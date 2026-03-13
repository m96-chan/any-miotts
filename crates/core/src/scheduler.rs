use std::collections::HashMap;
use std::time::Duration;

use tracing::info;

use crate::backend::{Backend, BenchmarkResult, ModelComponent};
use crate::device::DeviceKind;
use crate::error::TtsError;

/// Assignment of each model component to a specific backend index.
#[derive(Debug, Clone)]
pub struct Assignment {
    /// Maps each component to the index of the backend in the backends list.
    pub map: HashMap<ModelComponent, usize>,
}

/// Hard rules for component-device suitability.
///
/// Returns `true` if the given component is allowed on the given device kind.
fn is_suitable(component: ModelComponent, kind: DeviceKind) -> bool {
    match component {
        // LFM2 is autoregressive: GPU or CPU only, NPU unsuitable for variable-length loops.
        ModelComponent::Lfm2 => !kind.is_npu(),
        // MioCodec: fixed-shape single forward pass, runs anywhere.
        ModelComponent::MioCodec => true,
        // SpeakerEncoder: single forward pass, runs anywhere.
        ModelComponent::SpeakerEncoder => true,
    }
}

/// Preference score for a component on a given device kind (higher = more preferred).
/// Used as tie-breaker when no benchmark data is available.
fn preference_score(component: ModelComponent, kind: DeviceKind) -> u32 {
    match component {
        // LFM2: GPU strongly preferred for autoregressive generation speed.
        ModelComponent::Lfm2 => match kind {
            DeviceKind::CudaGpu | DeviceKind::MetalGpu => 100,
            DeviceKind::OpenClGpu | DeviceKind::VulkanGpu => 80,
            DeviceKind::Cpu => 50,
            _ => 10,
        },
        // MioCodec: NPU preferred (fixed-shape single forward pass), GPU also good.
        ModelComponent::MioCodec => match kind {
            DeviceKind::CoreMlAne | DeviceKind::QnnNpu => 100,
            DeviceKind::CudaGpu | DeviceKind::MetalGpu => 90,
            DeviceKind::OpenClGpu | DeviceKind::VulkanGpu => 70,
            DeviceKind::Cpu => 50,
        },
        // SpeakerEncoder: GPU preferred (one-time only, not latency-critical).
        ModelComponent::SpeakerEncoder => match kind {
            DeviceKind::CudaGpu | DeviceKind::MetalGpu => 100,
            DeviceKind::OpenClGpu | DeviceKind::VulkanGpu => 80,
            DeviceKind::CoreMlAne | DeviceKind::QnnNpu => 70,
            DeviceKind::Cpu => 50,
        },
    }
}

/// Run benchmark for a component on a backend, returning ops/sec.
///
/// Performs `warmup` warmup iterations followed by `measured` timed iterations.
/// Returns the average duration across measured iterations.
pub fn run_benchmark(
    backend: &dyn Backend,
    model: &dyn crate::backend::LoadedModel,
    component: ModelComponent,
    warmup: usize,
    measured: usize,
) -> Result<BenchmarkResult, TtsError> {
    use std::time::Instant;

    // Warmup iterations
    for _ in 0..warmup {
        backend.benchmark(model, component)?;
    }

    // Measured iterations
    let mut total = Duration::ZERO;
    for _ in 0..measured {
        let start = Instant::now();
        backend.benchmark(model, component)?;
        total += start.elapsed();
    }

    let avg_duration = total / measured as u32;

    Ok(BenchmarkResult {
        component,
        avg_duration,
        iterations: measured,
    })
}

/// Assign each model component to the best available backend.
///
/// Selection strategy:
/// 1. Filter backends by hard suitability rules
/// 2. If benchmark results are provided, pick the fastest
/// 3. Otherwise, pick by static preference score
pub fn auto_assign(backends: &[Box<dyn Backend>]) -> Result<Assignment, TtsError> {
    auto_assign_with_benchmarks(backends, &HashMap::new())
}

/// Assign with optional benchmark results.
///
/// `benchmarks` maps (backend_index, component) -> BenchmarkResult.
pub fn auto_assign_with_benchmarks(
    backends: &[Box<dyn Backend>],
    benchmarks: &HashMap<(usize, ModelComponent), BenchmarkResult>,
) -> Result<Assignment, TtsError> {
    let components = [
        ModelComponent::SpeakerEncoder,
        ModelComponent::Lfm2,
        ModelComponent::MioCodec,
    ];

    let mut map = HashMap::new();

    for &component in &components {
        // Find all candidate backends: must support the component and pass hard rules.
        let mut candidates: Vec<(usize, &dyn Backend)> = backends
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                b.supported_components().contains(&component)
                    && is_suitable(component, b.device_info().kind)
            })
            .map(|(i, b)| (i, b.as_ref()))
            .collect();

        if candidates.is_empty() {
            return Err(TtsError::NoBackend(component));
        }

        // Try to pick by benchmark results first.
        let best_idx = if let Some(best) = pick_by_benchmark(&candidates, component, benchmarks) {
            best
        } else {
            // Fall back to preference score.
            pick_by_preference(&mut candidates, component)
        };

        let backend = &backends[best_idx];
        info!(
            "Assigned {} -> {} ({}) [{}]",
            component,
            backend.name(),
            backend.device_info(),
            if benchmarks.contains_key(&(best_idx, component)) {
                "benchmark"
            } else {
                "preference"
            }
        );
        map.insert(component, best_idx);
    }

    Ok(Assignment { map })
}

/// Pick the best backend for a component based on benchmark results (lowest avg_duration).
fn pick_by_benchmark(
    candidates: &[(usize, &dyn Backend)],
    component: ModelComponent,
    benchmarks: &HashMap<(usize, ModelComponent), BenchmarkResult>,
) -> Option<usize> {
    let mut best: Option<(usize, Duration)> = None;
    for &(idx, _) in candidates {
        if let Some(result) = benchmarks.get(&(idx, component)) {
            match best {
                None => best = Some((idx, result.avg_duration)),
                Some((_, best_dur)) if result.avg_duration < best_dur => {
                    best = Some((idx, result.avg_duration));
                }
                _ => {}
            }
        }
    }
    best.map(|(idx, _)| idx)
}

/// Pick the best backend for a component based on static preference scores.
///
/// Tie-breaking: when two backends have the same preference score, prefer the
/// more specialized one (fewer supported components).  A backend that supports
/// only LFM2 via GGUF quantization is likely faster on CPU than a general
/// backend running full-precision safetensors.
fn pick_by_preference(
    candidates: &mut [(usize, &dyn Backend)],
    component: ModelComponent,
) -> usize {
    candidates.sort_by(|a, b| {
        let score_a = preference_score(component, a.1.device_info().kind);
        let score_b = preference_score(component, b.1.device_info().kind);
        score_b
            .cmp(&score_a)
            .then_with(|| {
                // Fewer supported components = more specialized = preferred
                a.1.supported_components()
                    .len()
                    .cmp(&b.1.supported_components().len())
            })
    });
    candidates[0].0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, ModelComponent};
    use crate::device::{DeviceInfo, DeviceKind};
    use std::time::Duration;

    /// A minimal mock backend for testing.
    struct MockBackend {
        name: String,
        device_info: DeviceInfo,
        components: Vec<ModelComponent>,
    }

    impl MockBackend {
        fn new(name: &str, kind: DeviceKind, components: Vec<ModelComponent>) -> Self {
            Self {
                name: name.to_string(),
                device_info: DeviceInfo::new(kind, name.to_string(), 0),
                components,
            }
        }
    }

    impl Backend for MockBackend {
        fn name(&self) -> &str {
            &self.name
        }
        fn supported_components(&self) -> &[ModelComponent] {
            &self.components
        }
        fn device_info(&self) -> &DeviceInfo {
            &self.device_info
        }
    }

    #[test]
    fn auto_assign_prefers_gpu_for_lfm2() {
        let backends: Vec<Box<dyn Backend>> = vec![
            Box::new(MockBackend::new(
                "cpu",
                DeviceKind::Cpu,
                vec![
                    ModelComponent::SpeakerEncoder,
                    ModelComponent::Lfm2,
                    ModelComponent::MioCodec,
                ],
            )),
            Box::new(MockBackend::new(
                "cuda",
                DeviceKind::CudaGpu,
                vec![
                    ModelComponent::SpeakerEncoder,
                    ModelComponent::Lfm2,
                    ModelComponent::MioCodec,
                ],
            )),
        ];

        let assignment = auto_assign(&backends).unwrap();
        // LFM2 should prefer CUDA GPU
        assert_eq!(assignment.map[&ModelComponent::Lfm2], 1);
        // SpeakerEncoder should also prefer GPU
        assert_eq!(assignment.map[&ModelComponent::SpeakerEncoder], 1);
    }

    #[test]
    fn auto_assign_lfm2_excludes_npu() {
        let backends: Vec<Box<dyn Backend>> = vec![
            Box::new(MockBackend::new(
                "npu",
                DeviceKind::CoreMlAne,
                vec![
                    ModelComponent::SpeakerEncoder,
                    ModelComponent::Lfm2,
                    ModelComponent::MioCodec,
                ],
            )),
            Box::new(MockBackend::new(
                "cpu",
                DeviceKind::Cpu,
                vec![
                    ModelComponent::SpeakerEncoder,
                    ModelComponent::Lfm2,
                    ModelComponent::MioCodec,
                ],
            )),
        ];

        let assignment = auto_assign(&backends).unwrap();
        // LFM2 must NOT be assigned to NPU
        assert_eq!(
            assignment.map[&ModelComponent::Lfm2],
            1,
            "LFM2 should be on CPU, not NPU"
        );
    }

    #[test]
    fn auto_assign_miocodec_prefers_npu() {
        let backends: Vec<Box<dyn Backend>> = vec![
            Box::new(MockBackend::new(
                "cpu",
                DeviceKind::Cpu,
                vec![
                    ModelComponent::SpeakerEncoder,
                    ModelComponent::Lfm2,
                    ModelComponent::MioCodec,
                ],
            )),
            Box::new(MockBackend::new(
                "npu",
                DeviceKind::CoreMlAne,
                vec![ModelComponent::SpeakerEncoder, ModelComponent::MioCodec],
            )),
        ];

        let assignment = auto_assign(&backends).unwrap();
        // MioCodec should prefer NPU
        assert_eq!(
            assignment.map[&ModelComponent::MioCodec],
            1,
            "MioCodec should prefer NPU"
        );
    }

    #[test]
    fn auto_assign_with_benchmark_overrides_preference() {
        let backends: Vec<Box<dyn Backend>> = vec![
            Box::new(MockBackend::new(
                "cpu",
                DeviceKind::Cpu,
                vec![
                    ModelComponent::SpeakerEncoder,
                    ModelComponent::Lfm2,
                    ModelComponent::MioCodec,
                ],
            )),
            Box::new(MockBackend::new(
                "cuda",
                DeviceKind::CudaGpu,
                vec![
                    ModelComponent::SpeakerEncoder,
                    ModelComponent::Lfm2,
                    ModelComponent::MioCodec,
                ],
            )),
        ];

        // Simulate: CPU is somehow faster for MioCodec
        let mut benchmarks = HashMap::new();
        benchmarks.insert(
            (0, ModelComponent::MioCodec),
            BenchmarkResult {
                component: ModelComponent::MioCodec,
                avg_duration: Duration::from_millis(10),
                iterations: 5,
            },
        );
        benchmarks.insert(
            (1, ModelComponent::MioCodec),
            BenchmarkResult {
                component: ModelComponent::MioCodec,
                avg_duration: Duration::from_millis(50),
                iterations: 5,
            },
        );

        let assignment = auto_assign_with_benchmarks(&backends, &benchmarks).unwrap();
        // Benchmark says CPU is faster for MioCodec
        assert_eq!(
            assignment.map[&ModelComponent::MioCodec],
            0,
            "Benchmark should override preference"
        );
    }

    #[test]
    fn auto_assign_no_backend_error() {
        let backends: Vec<Box<dyn Backend>> = vec![Box::new(MockBackend::new(
            "cpu",
            DeviceKind::Cpu,
            vec![ModelComponent::SpeakerEncoder],
        ))];

        let result = auto_assign(&backends);
        assert!(result.is_err());
    }

    #[test]
    fn hard_rules_suitability() {
        // LFM2 not suitable for NPU
        assert!(!is_suitable(ModelComponent::Lfm2, DeviceKind::CoreMlAne));
        assert!(!is_suitable(ModelComponent::Lfm2, DeviceKind::QnnNpu));
        // LFM2 suitable for GPU/CPU
        assert!(is_suitable(ModelComponent::Lfm2, DeviceKind::CudaGpu));
        assert!(is_suitable(ModelComponent::Lfm2, DeviceKind::MetalGpu));
        assert!(is_suitable(ModelComponent::Lfm2, DeviceKind::Cpu));
        // MioCodec runs anywhere
        assert!(is_suitable(ModelComponent::MioCodec, DeviceKind::CoreMlAne));
        assert!(is_suitable(ModelComponent::MioCodec, DeviceKind::CudaGpu));
        assert!(is_suitable(ModelComponent::MioCodec, DeviceKind::Cpu));
    }
}
