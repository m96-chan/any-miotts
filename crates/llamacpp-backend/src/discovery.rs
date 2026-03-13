//! Device discovery for the llama.cpp backend.
//!
//! Detects whether a Vulkan-capable GPU is available (e.g. Qualcomm Adreno
//! on Android) and falls back to CPU otherwise.

use any_miotts_core::device::{DeviceInfo, DeviceKind};

/// GPU detection result.
#[derive(Debug, Clone)]
pub struct GpuProbe {
    /// Whether a GPU was found.
    pub available: bool,
    /// Human-readable device name, if detected.
    pub device_name: Option<String>,
    /// Approximate global memory in bytes, if known.
    pub memory_bytes: Option<u64>,
}

/// Probe the system for a Vulkan GPU.
///
/// When the `vulkan` feature is enabled, this asks llama.cpp whether GPU
/// offload is available (i.e. whether the Vulkan GGML backend was compiled in
/// and a suitable device was found at runtime).
pub fn probe_gpu() -> GpuProbe {
    #[cfg(feature = "vulkan")]
    {
        use llama_cpp_2::llama_backend::LlamaBackend;
        if let Ok(backend) = LlamaBackend::init() {
            if backend.supports_gpu_offload() {
                return GpuProbe {
                    available: true,
                    device_name: Some("Vulkan GPU (llama.cpp)".into()),
                    memory_bytes: None,
                };
            }
        }
        return GpuProbe {
            available: false,
            device_name: None,
            memory_bytes: None,
        };
    }

    #[cfg(not(feature = "vulkan"))]
    GpuProbe {
        available: false,
        device_name: None,
        memory_bytes: None,
    }
}

/// Build a [`DeviceInfo`] for the llama.cpp backend.
///
/// Returns a Vulkan GPU device if detected, otherwise CPU.
pub fn discover_device() -> DeviceInfo {
    let probe = probe_gpu();
    if probe.available {
        let name = probe
            .device_name
            .unwrap_or_else(|| "Vulkan GPU".to_string());
        let mut info = DeviceInfo::new(DeviceKind::VulkanGpu, name, 0);
        if let Some(mem) = probe.memory_bytes {
            info = info.with_memory(mem);
        }
        info
    } else {
        let cpu_name = read_cpu_name();
        DeviceInfo::new(DeviceKind::Cpu, cpu_name, 0)
    }
}

/// Detect the optimal number of CPU threads for inference.
///
/// LLM single-token decode is memory-bandwidth bound, so using all cores
/// causes contention without improving throughput.  The sweet spot is
/// typically `num_performance_cores` on ARM big.LITTLE or `physical_cores`
/// on x86 (excluding hyper-threads).
///
/// Strategy:
/// 1. On Linux/Android: read sysfs cpufreq to count performance cores
/// 2. Fallback: `online_cpus / 2` (works for HT on x86 and big.LITTLE)
/// 3. Clamp to [1, 8]
pub fn optimal_thread_count() -> u32 {
    let count = detect_thread_count_inner();
    count.clamp(1, 8)
}

fn detect_thread_count_inner() -> u32 {
    // Try sysfs-based detection (Linux/Android)
    #[cfg(target_os = "linux")]
    {
        if let Some(n) = count_from_sysfs() {
            return n;
        }
    }

    // Fallback: online CPUs / 2
    #[cfg(target_os = "linux")]
    {
        if let Ok(online) = std::fs::read_to_string("/sys/devices/system/cpu/online") {
            if let Some(n) = parse_cpu_range_count(&online) {
                return (n / 2).max(1);
            }
        }
    }

    // Last resort
    4
}

/// Read max frequencies from sysfs and determine optimal threads.
///
/// On ARM big.LITTLE, the optimal count is the smaller of:
/// - Number of performance cores (above median frequency)
/// - Total cores / 2 (memory bandwidth saturation limit)
///
/// This handles Snapdragon 8 Elite (2 efficiency + 6 perf → returns 4)
/// and other big.LITTLE configs correctly.
#[cfg(target_os = "linux")]
fn count_from_sysfs() -> Option<u32> {
    let mut freqs = Vec::new();

    for i in 0..16u32 {
        let path = format!("/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq");
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(freq) = content.trim().parse::<u64>() {
                freqs.push(freq);
            }
        }
    }

    if freqs.is_empty() {
        return None;
    }

    let total = freqs.len() as u32;

    // Find the median frequency to separate efficiency vs performance cores
    let mut sorted = freqs.clone();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];

    // Performance cores = above median (for heterogeneous), or all (for homogeneous)
    let perf_cores = freqs.iter().filter(|&&f| f >= median).count() as u32;

    // Memory-bandwidth-bound workloads: cap at total/2
    let bandwidth_limit = (total / 2).max(1);

    Some(perf_cores.min(bandwidth_limit))
}

/// Parse a CPU range string like "0-7" and return the count.
#[cfg(target_os = "linux")]
fn parse_cpu_range_count(s: &str) -> Option<u32> {
    let s = s.trim();
    // Handle "0-7" format
    if let Some((start, end)) = s.split_once('-') {
        let start: u32 = start.parse().ok()?;
        let end: u32 = end.parse().ok()?;
        Some(end - start + 1)
    } else {
        // Single CPU "0"
        Some(1)
    }
}

/// Try to read the CPU model name from /proc/cpuinfo on Linux / Android.
fn read_cpu_name() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if let Some(rest) = line.strip_prefix("model name") {
                    if let Some(name) = rest.trim_start().strip_prefix(':') {
                        return name.trim().to_string();
                    }
                }
                // ARM/Android: look for "Hardware" line.
                if let Some(rest) = line.strip_prefix("Hardware") {
                    if let Some(name) = rest.trim_start().strip_prefix(':') {
                        return name.trim().to_string();
                    }
                }
            }
        }
    }
    "CPU".to_string()
}
