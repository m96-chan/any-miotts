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
