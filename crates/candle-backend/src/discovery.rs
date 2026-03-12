use any_miotts_core::device::{DeviceInfo, DeviceKind};
use candle_core::Device;

/// Discover available candle devices on this system.
///
/// Returns devices ordered by preference: CUDA GPUs first, then Metal, then CPU.
/// Each entry includes richer device info (name, memory) when available.
#[allow(clippy::vec_init_then_push)]
pub fn discover_devices() -> Vec<(DeviceInfo, Device)> {
    let mut devices = Vec::new();

    // Try CUDA GPUs
    #[cfg(feature = "cuda")]
    {
        for i in 0..8 {
            match Device::new_cuda(i) {
                Ok(dev) => {
                    let name = cuda_device_name(i).unwrap_or_else(|| format!("CUDA:{i}"));
                    let mut info = DeviceInfo::new(DeviceKind::CudaGpu, name, i);
                    if let Some(mem) = cuda_device_memory(i) {
                        info = info.with_memory(mem);
                    }
                    devices.push((info, dev));
                }
                Err(_) => break,
            }
        }
    }

    // Try Metal (macOS/iOS)
    #[cfg(feature = "metal")]
    {
        if let Ok(dev) = Device::new_metal(0) {
            let info = DeviceInfo::new(DeviceKind::MetalGpu, "Metal GPU".to_string(), 0);
            devices.push((info, dev));
        }
    }

    // Always include CPU as fallback
    let cpu_name = cpu_name();
    devices.push((DeviceInfo::new(DeviceKind::Cpu, cpu_name, 0), Device::Cpu));

    devices
}

/// Try to read the CPU model name from /proc/cpuinfo on Linux.
fn cpu_name() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(cpuinfo) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in cpuinfo.lines() {
                if let Some(name) = line.strip_prefix("model name") {
                    if let Some(name) = name.trim_start().strip_prefix(':') {
                        return name.trim().to_string();
                    }
                }
            }
        }
    }
    "CPU".to_string()
}

/// Try to get the CUDA device name via nvidia-smi.
#[cfg(feature = "cuda")]
fn cuda_device_name(ordinal: usize) -> Option<String> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=name",
            "--format=csv,noheader",
            &format!("--id={ordinal}"),
        ])
        .output()
        .ok()?;
    if output.status.success() {
        let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !name.is_empty() {
            return Some(name);
        }
    }
    None
}

/// Try to get the CUDA device total memory in bytes via nvidia-smi.
#[cfg(feature = "cuda")]
fn cuda_device_memory(ordinal: usize) -> Option<u64> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
            &format!("--id={ordinal}"),
        ])
        .output()
        .ok()?;
    if output.status.success() {
        let s = String::from_utf8_lossy(&output.stdout).trim().to_string();
        // nvidia-smi reports in MiB
        s.parse::<u64>().ok().map(|mib| mib * 1024 * 1024)
    } else {
        None
    }
}
