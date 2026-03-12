//! Hexagon NPU detection.
//!
//! On Android, checks whether a Qualcomm Hexagon NPU is available by probing
//! for QNN delegate libraries and SoC properties.  On other platforms, always
//! reports the NPU as unavailable.

use any_miotts_core::device::{DeviceInfo, DeviceKind};

/// Result of probing for the Qualcomm Hexagon NPU.
#[derive(Debug, Clone)]
pub struct HexagonProbe {
    /// Whether a Hexagon NPU was detected.
    pub available: bool,
    /// Human-readable SoC / NPU description, if detected.
    pub device_name: Option<String>,
}

/// Probe the system for a Qualcomm Hexagon NPU.
///
/// On Android, checks for the QNN HTP shared library and reads SoC
/// information from system properties.
pub fn probe_hexagon() -> HexagonProbe {
    #[cfg(target_os = "android")]
    {
        return probe_hexagon_android();
    }

    #[cfg(not(target_os = "android"))]
    {
        HexagonProbe {
            available: false,
            device_name: None,
        }
    }
}

/// Build a [`DeviceInfo`] for the QNN/Hexagon backend.
///
/// Returns an NPU device if detected, otherwise CPU.
pub fn discover_device() -> DeviceInfo {
    let probe = probe_hexagon();
    if probe.available {
        let name = probe
            .device_name
            .unwrap_or_else(|| "Qualcomm Hexagon NPU".to_string());
        DeviceInfo::new(DeviceKind::QnnNpu, name, 0)
    } else {
        DeviceInfo::new(DeviceKind::Cpu, "CPU (no Hexagon NPU)".to_string(), 0)
    }
}

/// Android-specific Hexagon NPU detection.
#[cfg(target_os = "android")]
fn probe_hexagon_android() -> HexagonProbe {
    use std::path::Path;

    // Check for QNN HTP delegate library
    let qnn_lib_paths = [
        "/data/local/tmp/libQnnHtp.so",
        "/vendor/lib64/libQnnHtp.so",
        "/system/vendor/lib64/libQnnHtp.so",
    ];

    let qnn_available = qnn_lib_paths.iter().any(|p| Path::new(p).exists());

    if !qnn_available {
        return HexagonProbe {
            available: false,
            device_name: None,
        };
    }

    // Try to read SoC name from Android system properties
    let soc_name = read_android_property("ro.soc.model")
        .or_else(|| read_android_property("ro.board.platform"))
        .map(|soc| format!("Qualcomm {soc} Hexagon NPU"));

    HexagonProbe {
        available: true,
        device_name: soc_name,
    }
}

/// Read an Android system property via getprop.
#[cfg(target_os = "android")]
fn read_android_property(prop: &str) -> Option<String> {
    std::process::Command::new("getprop")
        .arg(prop)
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                let val = String::from_utf8(output.stdout)
                    .ok()
                    .map(|s| s.trim().to_string());
                val.filter(|s| !s.is_empty())
            } else {
                None
            }
        })
}
