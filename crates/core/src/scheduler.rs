use std::collections::HashMap;

use tracing::info;

use crate::backend::{Backend, ModelComponent};
use crate::error::TtsError;

/// Assignment of each model component to a specific backend index.
#[derive(Debug, Clone)]
pub struct Assignment {
    /// Maps each component to the index of the backend in the backends list.
    pub map: HashMap<ModelComponent, usize>,
}

/// Assign each model component to the best available backend.
///
/// For Phase 1, this simply picks the first backend that supports each component.
/// Phase 2 will add benchmark-based selection.
pub fn auto_assign(backends: &[Box<dyn Backend>]) -> Result<Assignment, TtsError> {
    let components = [
        ModelComponent::SpeakerEncoder,
        ModelComponent::Lfm2,
        ModelComponent::MioCodec,
    ];

    let mut map = HashMap::new();

    for &component in &components {
        let mut assigned = false;
        for (i, backend) in backends.iter().enumerate() {
            if backend.supported_components().contains(&component) {
                info!(
                    "Assigned {} → {} ({})",
                    component,
                    backend.name(),
                    backend.device_info()
                );
                map.insert(component, i);
                assigned = true;
                break;
            }
        }
        if !assigned {
            return Err(TtsError::NoBackend(component));
        }
    }

    Ok(Assignment { map })
}
