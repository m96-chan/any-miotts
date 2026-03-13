//! Debug LFM2 inference with detailed per-step timing.
//!
//! Tests different GPU_LAYERS values to find the bottleneck.
//!
//! Usage:
//!   MIOTTS_GPU_LAYERS=0 cargo run -p any-miotts --features llamacpp-vulkan --example debug_vulkan
//!   MIOTTS_GPU_LAYERS=8 cargo run -p any-miotts --features llamacpp-vulkan --example debug_vulkan
//!   MIOTTS_GPU_LAYERS=99 cargo run -p any-miotts --features llamacpp-vulkan --example debug_vulkan

use std::path::PathBuf;
use std::time::Instant;

fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "debug,llama_cpp_2=info".into()),
        )
        .init();

    let gpu_layers: u32 = std::env::var("MIOTTS_GPU_LAYERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(99);

    let gguf_path = std::env::var("MIOTTS_GGUF_PATH").unwrap_or_else(|_| {
        "/home/m96-chan/project/m96-chan/any-miotts/models/gguf/MioTTS-2.6B-Q4_K_M.gguf".into()
    });

    tracing::info!("=== LFM2 Vulkan Debug ===");
    tracing::info!("GGUF: {gguf_path}");
    tracing::info!("GPU_LAYERS: {gpu_layers}");

    // Auto-detect optimal thread count, allow env override
    let auto_threads = any_miotts::llamacpp_discovery::optimal_thread_count();
    let n_threads = std::env::var("MIOTTS_THREADS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(auto_threads);
    tracing::info!("Threads: {} (auto-detected: {})", n_threads, auto_threads);

    // Initialize llama.cpp backend directly
    let config = any_miotts::LlamaCppConfig {
        n_gpu_layers: gpu_layers,
        n_ctx: 4096,
        n_batch: 512,
        n_threads,
        gguf_path: Some(PathBuf::from(&gguf_path)),
    };

    let device_info = any_miotts::llamacpp_discovery::discover_device();
    tracing::info!("Device: {device_info}");

    let backend = any_miotts::LlamaCppBackend::new(device_info, config);

    use any_miotts::Backend;

    // Load model
    let t0 = Instant::now();
    let model = backend
        .load_model(
            any_miotts::ModelComponent::Lfm2,
            std::path::Path::new(""),
        )
        .expect("Failed to load model");
    tracing::info!("Model loaded in {:.2}s", t0.elapsed().as_secs_f64());

    // Simulate a typical TTS prompt (short text → ~10-30 tokens)
    // Use simple token IDs that won't cause issues
    let prompt_tokens: Vec<u32> = vec![1, 100, 200, 300, 400, 500, 600, 700, 800, 900];
    tracing::info!("Prompt tokens: {} tokens", prompt_tokens.len());

    // Prefill
    let t_prefill = Instant::now();
    let (mut state, logits) = backend
        .lfm2_prefill(model.as_ref(), &prompt_tokens)
        .expect("Prefill failed");
    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
    tracing::info!(
        "Prefill: {:.1}ms for {} tokens ({:.1} tok/s)",
        prefill_ms,
        prompt_tokens.len(),
        prompt_tokens.len() as f64 / prefill_ms * 1000.0,
    );

    // Find the max logit token to use as input for decode steps
    let mut next_token = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i as u32)
        .unwrap_or(1);

    // Decode steps with per-step timing
    let max_steps = 200;
    let mut step_times_ms = Vec::with_capacity(max_steps);
    let mut total_tokens = 0u32;

    tracing::info!("Starting decode loop (max {} steps)...", max_steps);
    let t_decode_start = Instant::now();

    for step in 0..max_steps {
        let t_step = Instant::now();
        let step_logits = match backend.lfm2_decode_step(model.as_ref(), state.as_mut(), next_token)
        {
            Ok(l) => l,
            Err(e) => {
                tracing::error!("Decode step {step} failed: {e}");
                break;
            }
        };
        let step_ms = t_step.elapsed().as_secs_f64() * 1000.0;
        step_times_ms.push(step_ms);
        total_tokens += 1;

        // Greedy next token
        next_token = step_logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(1);

        // Log every 10th step
        if (step + 1) % 10 == 0 {
            let elapsed = t_decode_start.elapsed().as_secs_f64();
            tracing::info!(
                "  step {}: {:.2}ms (avg {:.2}ms/tok, {:.1} tok/s)",
                step + 1,
                step_ms,
                elapsed * 1000.0 / total_tokens as f64,
                total_tokens as f64 / elapsed,
            );
        }

        // Check for EOS (token 0 or whatever the model uses)
        // Don't stop early for benchmarking purposes, just note it
        if next_token == 0 || next_token == 2 {
            tracing::warn!("EOS token {} at step {}", next_token, step + 1);
            // Continue anyway for timing purposes
        }
    }

    let total_decode_ms = t_decode_start.elapsed().as_secs_f64() * 1000.0;

    // Statistics
    tracing::info!("=== Results (GPU_LAYERS={gpu_layers}) ===");
    tracing::info!(
        "Decode: {} tokens in {:.1}ms ({:.1} tok/s)",
        total_tokens,
        total_decode_ms,
        total_tokens as f64 / total_decode_ms * 1000.0,
    );

    if !step_times_ms.is_empty() {
        let mut sorted = step_times_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let median = sorted[sorted.len() / 2];
        let p95 = sorted[(sorted.len() as f64 * 0.95) as usize];
        let mean = step_times_ms.iter().sum::<f64>() / step_times_ms.len() as f64;

        tracing::info!("Per-step timing (ms):");
        tracing::info!("  min={:.2} median={:.2} mean={:.2} p95={:.2} max={:.2}", min, median, mean, p95, max);

        // First 5 steps (often slower due to warmup)
        let first5: Vec<String> = step_times_ms.iter().take(5).map(|t| format!("{:.1}", t)).collect();
        tracing::info!("  first 5 steps: [{}]", first5.join(", "));

        // Variance analysis: are there periodic spikes?
        let threshold = mean * 2.0;
        let spikes: Vec<(usize, f64)> = step_times_ms
            .iter()
            .enumerate()
            .filter(|(_, &t)| t > threshold)
            .map(|(i, &t)| (i, t))
            .collect();
        if !spikes.is_empty() {
            tracing::warn!("{} spike(s) > {:.1}ms (2x mean):", spikes.len(), threshold);
            for (i, t) in spikes.iter().take(10) {
                tracing::warn!("  step {}: {:.2}ms", i, t);
            }
        }
    }
}
