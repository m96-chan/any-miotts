//! Disk cache for speaker embeddings.
//!
//! Avoids recomputing the speaker embedding (WavLM + GlobalEncoder) when the
//! same reference WAV file is used again.  The cache key is derived from the
//! file path, size and modification time – cheap to compute and good enough for
//! local development / on-device usage.

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use tracing::info;

use crate::backend::TensorData;
use crate::error::TtsError;

/// Magic bytes for the cache file format.
const MAGIC: &[u8; 4] = b"SPKE";
/// Current cache format version.
const VERSION: u32 = 1;

/// Speaker embedding disk cache.
pub struct SpeakerCache {
    cache_dir: PathBuf,
}

impl SpeakerCache {
    pub fn new(cache_dir: PathBuf) -> Self {
        Self { cache_dir }
    }

    /// Derive a cache key from the reference WAV file metadata.
    ///
    /// Uses file name + size + mtime to avoid reading the whole file.
    fn cache_key(wav_path: &Path) -> Result<String, TtsError> {
        let meta = fs::metadata(wav_path)
            .map_err(|e| TtsError::Model(format!("Cache key metadata: {e}")))?;
        let size = meta.len();
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let name = wav_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();
        Ok(format!("{name}_{size}_{mtime}"))
    }

    /// Cache file path for a given key.
    fn cache_path(&self, key: &str) -> PathBuf {
        self.cache_dir.join(format!("{key}.spk_embed"))
    }

    /// Try to load a cached speaker embedding.
    pub fn load(&self, wav_path: &Path) -> Result<Option<TensorData>, TtsError> {
        let key = Self::cache_key(wav_path)?;
        let path = self.cache_path(&key);

        if !path.exists() {
            return Ok(None);
        }

        let mut file = fs::File::open(&path)
            .map_err(|e| TtsError::Model(format!("Open cache: {e}")))?;

        // Read and validate header
        let mut magic = [0u8; 4];
        let mut ver_buf = [0u8; 4];
        let mut dim_buf = [0u8; 4];
        file.read_exact(&mut magic)
            .and_then(|_| file.read_exact(&mut ver_buf))
            .and_then(|_| file.read_exact(&mut dim_buf))
            .map_err(|e| TtsError::Model(format!("Read cache header: {e}")))?;

        if &magic != MAGIC {
            info!("Cache file has invalid magic, recomputing");
            return Ok(None);
        }
        let version = u32::from_le_bytes(ver_buf);
        if version != VERSION {
            info!("Cache file version mismatch ({version} != {VERSION}), recomputing");
            return Ok(None);
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;

        // Read f32 data
        let mut data_bytes = vec![0u8; dim * 4];
        file.read_exact(&mut data_bytes)
            .map_err(|e| TtsError::Model(format!("Read cache data: {e}")))?;

        let data: Vec<f32> = data_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        info!(
            "Loaded speaker embedding from cache ({} dims, {})",
            dim,
            path.display()
        );

        Ok(Some(TensorData::F32 {
            data,
            shape: vec![dim],
        }))
    }

    /// Save a speaker embedding to the cache.
    pub fn save(&self, wav_path: &Path, embedding: &TensorData) -> Result<(), TtsError> {
        let data = match embedding {
            TensorData::F32 { data, .. } => data,
            _ => {
                info!("Cannot cache non-F32 speaker embedding, skipping");
                return Ok(());
            }
        };

        let key = Self::cache_key(wav_path)?;
        let path = self.cache_path(&key);

        // Ensure cache directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| TtsError::Model(format!("Create cache dir: {e}")))?;
        }

        let mut file = fs::File::create(&path)
            .map_err(|e| TtsError::Model(format!("Create cache file: {e}")))?;

        // Write header
        file.write_all(MAGIC)
            .and_then(|_| file.write_all(&VERSION.to_le_bytes()))
            .and_then(|_| file.write_all(&(data.len() as u32).to_le_bytes()))
            .map_err(|e| TtsError::Model(format!("Write cache header: {e}")))?;

        // Write f32 data
        for &val in data {
            file.write_all(&val.to_le_bytes())
                .map_err(|e| TtsError::Model(format!("Write cache data: {e}")))?;
        }

        info!(
            "Saved speaker embedding to cache ({} dims, {})",
            data.len(),
            path.display()
        );
        Ok(())
    }
}
