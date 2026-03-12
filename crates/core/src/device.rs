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

impl DeviceKind {
    /// Whether this device kind is a GPU.
    pub fn is_gpu(&self) -> bool {
        matches!(
            self,
            Self::CudaGpu | Self::MetalGpu | Self::OpenClGpu | Self::VulkanGpu
        )
    }

    /// Whether this device kind is an NPU / accelerator.
    pub fn is_npu(&self) -> bool {
        matches!(self, Self::CoreMlAne | Self::QnnNpu)
    }
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
    /// Total device memory in bytes, if known.
    pub memory_bytes: Option<u64>,
}

impl DeviceInfo {
    /// Create a new DeviceInfo with no memory info.
    pub fn new(kind: DeviceKind, name: String, index: usize) -> Self {
        Self {
            kind,
            name,
            index,
            memory_bytes: None,
        }
    }

    /// Builder-style setter for memory.
    pub fn with_memory(mut self, bytes: u64) -> Self {
        self.memory_bytes = Some(bytes);
        self
    }

    /// Memory in MiB, if known.
    pub fn memory_mib(&self) -> Option<u64> {
        self.memory_bytes.map(|b| b / (1024 * 1024))
    }
}

impl fmt::Display for DeviceInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]: {}", self.kind, self.index, self.name)?;
        if let Some(mib) = self.memory_mib() {
            write!(f, " ({mib} MiB)")?;
        }
        Ok(())
    }
}
