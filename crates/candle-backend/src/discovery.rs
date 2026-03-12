use any_miotts_core::device::{DeviceInfo, DeviceKind};
use candle_core::Device;

/// Discover available candle devices on this system.
#[allow(clippy::vec_init_then_push)]
pub fn discover_devices() -> Vec<(DeviceInfo, Device)> {
    let mut devices = Vec::new();

    // Try CUDA GPUs
    #[cfg(feature = "cuda")]
    {
        // Try up to 8 CUDA devices
        for i in 0..8 {
            match Device::new_cuda(i) {
                Ok(dev) => {
                    devices.push((
                        DeviceInfo {
                            kind: DeviceKind::CudaGpu,
                            name: format!("CUDA:{i}"),
                            index: i,
                        },
                        dev,
                    ));
                }
                Err(_) => break,
            }
        }
    }

    // Try Metal (macOS/iOS)
    #[cfg(feature = "metal")]
    {
        if let Ok(dev) = Device::new_metal(0) {
            devices.push((
                DeviceInfo {
                    kind: DeviceKind::MetalGpu,
                    name: "Metal GPU".to_string(),
                    index: 0,
                },
                dev,
            ));
        }
    }

    // Always include CPU as fallback
    devices.push((
        DeviceInfo {
            kind: DeviceKind::Cpu,
            name: "CPU".to_string(),
            index: 0,
        },
        Device::Cpu,
    ));

    devices
}
