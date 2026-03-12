//! ANE (Apple Neural Engine) detection.
//!
//! On macOS / iOS, checks whether the device has an ANE available.
//! On other platforms, always reports ANE as unavailable.

use any_miotts_core::device::{DeviceInfo, DeviceKind};

/// Result of probing for the Apple Neural Engine.
#[derive(Debug, Clone)]
pub struct AneProbe {
    /// Whether an ANE was detected.
    pub available: bool,
    /// Human-readable device description, if detected.
    pub device_name: Option<String>,
}

/// Probe the system for an Apple Neural Engine.
///
/// On Apple platforms, CoreML automatically routes work to the ANE when
/// available.  We report it as available on any Apple Silicon device.
pub fn probe_ane() -> AneProbe {
    #[cfg(target_vendor = "apple")]
    {
        // On Apple Silicon (M-series or A-series), the ANE is always present.
        // CoreML's MLModel will automatically use it for compatible models.
        // We detect Apple Silicon via the architecture.
        #[cfg(target_arch = "aarch64")]
        {
            return AneProbe {
                available: true,
                device_name: Some(read_apple_chip_name()),
            };
        }

        // Intel Macs do not have an ANE.
        #[cfg(not(target_arch = "aarch64"))]
        {
            return AneProbe {
                available: false,
                device_name: None,
            };
        }
    }

    #[cfg(not(target_vendor = "apple"))]
    {
        AneProbe {
            available: false,
            device_name: None,
        }
    }
}

/// Build a [`DeviceInfo`] for the CoreML/ANE backend.
///
/// Returns an ANE device if detected, otherwise CPU (though the backend
/// will still fail with `Unsupported` on non-Apple platforms).
pub fn discover_device() -> DeviceInfo {
    let probe = probe_ane();
    if probe.available {
        let name = probe
            .device_name
            .unwrap_or_else(|| "Apple Neural Engine".to_string());
        DeviceInfo::new(DeviceKind::CoreMlAne, name, 0)
    } else {
        DeviceInfo::new(DeviceKind::Cpu, "CPU (no ANE)".to_string(), 0)
    }
}

/// Read the Apple chip name via sysctl on macOS.
#[cfg(target_vendor = "apple")]
fn read_apple_chip_name() -> String {
    // Try to read the chip name from sysctl.
    // `sysctl -n machdep.cpu.brand_string` on macOS.
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                String::from_utf8(output.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "Apple Silicon ANE".to_string())
}
