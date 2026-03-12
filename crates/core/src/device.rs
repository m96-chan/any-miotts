use std::fmt;

/// The kind of compute device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceKind {
    Cpu,
    CudaGpu,
    MetalGpu,
    OpenClGpu,
    VulkanGpu,
    CoreMlAne,
    QnnNpu,
}

impl fmt::Display for DeviceKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::CudaGpu => write!(f, "CUDA GPU"),
            Self::MetalGpu => write!(f, "Metal GPU"),
            Self::OpenClGpu => write!(f, "OpenCL GPU"),
            Self::VulkanGpu => write!(f, "Vulkan GPU"),
            Self::CoreMlAne => write!(f, "CoreML ANE"),
            Self::QnnNpu => write!(f, "QNN NPU"),
        }
    }
}

/// Information about a discovered compute device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device kind.
    pub kind: DeviceKind,
    /// Human-readable name (e.g. "NVIDIA RTX 4090", "Apple M2 GPU").
    pub name: String,
    /// Device index (for multi-GPU systems).
    pub index: usize,
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]: {}", self.kind, self.index, self.name)
    }
}
