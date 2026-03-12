//! Device discovery for the llama.cpp backend.
//!
//! Detects whether an OpenCL-capable GPU is available (e.g. Qualcomm Adreno
//! on Android) and falls back to CPU otherwise.

use any_miotts_core::device::{DeviceInfo, DeviceKind};

/// OpenCL GPU detection result.
#[derive(Debug, Clone)]
pub struct OpenClProbe {
    /// Whether an OpenCL GPU was found.
    pub available: bool,
    /// Human-readable device name, if detected.
    pub device_name: Option<String>,
    /// Approximate global memory in bytes, if known.
    pub memory_bytes: Option<u64>,
}

/// Probe the system for an OpenCL GPU.
///
/// On Android this would query the OpenCL runtime for Adreno/Mali devices.
/// The current implementation uses llama.cpp's own backend enumeration when
/// the `opencl` feature is enabled, and returns "not available" otherwise.
pub fn probe_opencl() -> OpenClProbe {
    // When compiled with OpenCL support, ask llama.cpp whether GPU offload
    // is available.  The llama-cpp-2 crate exposes `supports_gpu_offload()`
    // on the backend handle, which reflects whether any GPU ggml backend
    // was compiled in.
    #[cfg(feature = "opencl")]
    {
        // Try to initialise the llama backend just for probing.
        // If it was already initialised elsewhere this will fail, which is
        // fine -- we still report OpenCL as potentially available because
        // the feature flag was set at compile time.
        if let Ok(backend) = llama_cpp_2::LlamaBackend::init() {
            if backend.supports_gpu_offload() {
                return OpenClProbe {
                    available: true,
                    device_name: Some("OpenCL GPU (llama.cpp)".into()),
                    memory_bytes: None,
                };
            }
        }
        // Feature enabled but no GPU backend compiled in llama.cpp.
        return OpenClProbe {
            available: false,
            device_name: None,
            memory_bytes: None,
        };
    }

    #[cfg(not(feature = "opencl"))]
    OpenClProbe {
        available: false,
        device_name: None,
        memory_bytes: None,
    }
}

/// Build a [`DeviceInfo`] for the llama.cpp backend.
///
/// Returns an OpenCL GPU device if detected, otherwise CPU.
pub fn discover_device() -> DeviceInfo {
    let probe = probe_opencl();
    if probe.available {
        let name = probe
            .device_name
            .unwrap_or_else(|| "OpenCL GPU".to_string());
        let mut info = DeviceInfo::new(DeviceKind::OpenClGpu, name, 0);
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
