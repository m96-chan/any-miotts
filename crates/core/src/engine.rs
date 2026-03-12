use std::collections::HashMap;
use std::path::Path;

use tracing::{debug, info};

use crate::backend::{Backend, LoadedModel, ModelComponent, TensorData};
use crate::error::TtsError;
use crate::sampling::{self, GenerateParams};
use crate::scheduler::{self, Assignment};

/// Unified TTS engine that dispatches to the best backend per component.
pub struct TtsEngine {
    backends: Vec<Box<dyn Backend>>,
    assignment: Assignment,
    models: HashMap<ModelComponent, Box<dyn LoadedModel>>,
    /// Tokenizer for LFM2 prompt encoding / codec token extraction.
    tokenizer: tokenizers::Tokenizer,
    /// Cached speaker embedding from reference wav.
    speaker_embedding: TensorData,
    /// LFM2 config: eos_token_id.
    eos_token_id: u32,
    /// Output sample rate from MioCodec.
    sample_rate: usize,
}

impl TtsEngine {
    /// Build a TtsEngine from a set of backends.
    ///
    /// 1. Auto-assign components to backends
    /// 2. Ensure models are downloaded
    /// 3. Load models
    /// 4. Compute speaker embedding from reference wav
    pub fn build(
        backends: Vec<Box<dyn Backend>>,
        reference_wav: &Path,
        tokenizer: tokenizers::Tokenizer,
        eos_token_id: u32,
        sample_rate: usize,
    ) -> Result<Self, TtsError> {
        let assignment = scheduler::auto_assign(&backends)?;

        // Load models for each assigned component
        let mut models = HashMap::new();
        for (&component, &backend_idx) in &assignment.map {
            let backend = &backends[backend_idx];
            info!("Loading {} on {}...", component, backend.name());
            // For Phase 1, model_dir is unused; CandleBackend handles download internally
            let model = backend.load_model(component, Path::new(""))?;
            models.insert(component, model);
        }

        // Compute speaker embedding
        info!(
            "Computing speaker embedding from {}...",
            reference_wav.display()
        );
        let spk_backend_idx = assignment.map[&ModelComponent::SpeakerEncoder];
        let spk_model = models
            .get(&ModelComponent::SpeakerEncoder)
            .ok_or(TtsError::NoBackend(ModelComponent::SpeakerEncoder))?;

        let wav_data = load_and_resample_wav(reference_wav)?;
        let speaker_embedding =
            backends[spk_backend_idx].encode_speaker(spk_model.as_ref(), &wav_data)?;
        info!("Speaker embedding computed");

        Ok(Self {
            backends,
            assignment,
            models,
            tokenizer,
            speaker_embedding,
            eos_token_id,
            sample_rate,
        })
    }

    /// Output sample rate in Hz.
    pub fn sample_rate(&self) -> usize {
        self.sample_rate
    }

    /// Access the tokenizer.
    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Run a short warmup forward pass to JIT-compile GPU kernels.
    pub fn warmup(&self) -> Result<(), TtsError> {
        let prompt = sampling::build_prompt("a");
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| TtsError::Inference(format!("Tokenize: {e}")))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();

        let lfm2_idx = self.assignment.map[&ModelComponent::Lfm2];
        let lfm2_model = self.models.get(&ModelComponent::Lfm2).unwrap();
        let (_state, _logits) =
            self.backends[lfm2_idx].lfm2_prefill(lfm2_model.as_ref(), &input_ids)?;

        Ok(())
    }

    /// Synthesize speech from text. Returns PCM f32 samples at the codec's sample rate.
    pub fn synthesize(&self, text: &str) -> Result<Vec<f32>, TtsError> {
        // Step 1: Build prompt and tokenize
        let prompt = sampling::build_prompt(text);
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| TtsError::Inference(format!("Tokenize: {e}")))?;
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        info!("Tokenized '{}' -> {} tokens", text, input_ids.len());

        // Step 2: Generate codec tokens with LFM2
        let params = GenerateParams {
            eos_token_id: self.eos_token_id,
            ..GenerateParams::default()
        };
        let generated = self.generate_tokens(&input_ids, &params)?;

        // Step 3: Extract codec token indices
        let codec_tokens = sampling::extract_codec_tokens(&generated, &|id| {
            self.tokenizer.id_to_token(id)
        });
        info!(
            "Generated {} codec tokens from {} raw tokens",
            codec_tokens.len(),
            generated.len()
        );

        if codec_tokens.is_empty() {
            return Err(TtsError::Inference("No codec tokens generated".into()));
        }

        // Step 4: Decode with MioCodec
        let t_miocodec = std::time::Instant::now();
        let miocodec_idx = self.assignment.map[&ModelComponent::MioCodec];
        let miocodec_model = self.models.get(&ModelComponent::MioCodec).unwrap();
        let pcm = self.backends[miocodec_idx].miocodec_decode(
            miocodec_model.as_ref(),
            &codec_tokens,
            &self.speaker_embedding,
        )?;
        let miocodec_ms = t_miocodec.elapsed().as_secs_f64() * 1000.0;
        info!("MioCodec: {miocodec_ms:.1}ms ({} samples)", pcm.len());

        // Peak-normalize to -1dBFS (0.89)
        let peak = pcm.iter().copied().fold(0.0f32, |a, x| a.max(x.abs()));
        let pcm = if peak > 1e-6 {
            let gain = 0.89 / peak;
            pcm.into_iter().map(|x| x * gain).collect()
        } else {
            pcm
        };

        let rms = (pcm.iter().map(|x| x * x).sum::<f32>() / pcm.len() as f32).sqrt();
        info!(
            "Synthesized {} samples ({:.2}s): peak_raw={peak:.4}, rms={rms:.4}",
            pcm.len(),
            pcm.len() as f32 / self.sample_rate as f32
        );
        Ok(pcm)
    }

    /// Autoregressive token generation using LFM2 backend.
    fn generate_tokens(
        &self,
        input_ids: &[u32],
        params: &GenerateParams,
    ) -> Result<Vec<u32>, TtsError> {
        let lfm2_idx = self.assignment.map[&ModelComponent::Lfm2];
        let backend = &self.backends[lfm2_idx];
        let model = self.models.get(&ModelComponent::Lfm2).unwrap();

        // Prefill
        let t_prefill = std::time::Instant::now();
        let (mut state, mut logits) = backend.lfm2_prefill(model.as_ref(), input_ids)?;
        info!(
            "Prefill: {} tokens in {:.1}ms",
            input_ids.len(),
            t_prefill.elapsed().as_secs_f64() * 1000.0
        );

        if tracing::enabled!(tracing::Level::DEBUG) {
            let mut indexed: Vec<(usize, f32)> =
                logits.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top5: Vec<String> = indexed[..5.min(indexed.len())]
                .iter()
                .map(|(i, v)| format!("{i}={v:.3}"))
                .collect();
            debug!("Prefill logits top5: [{}]", top5.join(", "));
        }

        let mut generated = Vec::with_capacity(params.max_tokens);
        let gen_start = std::time::Instant::now();

        for step in 0..params.max_tokens {
            let next_token = sampling::sample_logits(&logits, params);

            if next_token == params.eos_token_id {
                info!("EOS at step {step}");
                break;
            }
            generated.push(next_token);

            if step < 3 || step % 100 == 99 {
                let elapsed = gen_start.elapsed().as_secs_f32();
                info!(
                    "  step {}: tok/s = {:.1}, elapsed = {:.1}ms, last_id = {next_token}",
                    step + 1,
                    (step + 1) as f32 / elapsed,
                    elapsed * 1000.0
                );
            }

            logits = backend.lfm2_decode_step(model.as_ref(), state.as_mut(), next_token)?;
        }

        let elapsed = gen_start.elapsed().as_secs_f32();
        info!(
            "Generated {} tokens in {:.2}s ({:.1} tok/s)",
            generated.len(),
            elapsed,
            generated.len() as f32 / elapsed
        );
        Ok(generated)
    }
}

/// Load a WAV file and resample to 16kHz mono f32 PCM.
fn load_and_resample_wav(path: &Path) -> Result<Vec<f32>, TtsError> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| TtsError::Model(format!("Read reference wav: {e}")))?;
    let spec = reader.spec();
    let src_rate = spec.sample_rate;
    let src_channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1u32 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max_val)
                .collect()
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
    };

    // Downmix to mono
    let mono = if src_channels <= 1 {
        samples
    } else {
        let mut out = Vec::with_capacity(samples.len() / src_channels);
        for frame in samples.chunks(src_channels) {
            let sum: f32 = frame.iter().sum();
            out.push(sum / src_channels as f32);
        }
        out
    };

    // Resample to 16kHz
    let target_rate = 16000u32;
    if src_rate == target_rate {
        Ok(mono)
    } else {
        Ok(resample_sinc(&mono, src_rate, target_rate))
    }
}

/// Windowed sinc resampling (matches torchaudio.transforms.Resample defaults).
fn resample_sinc(input: &[f32], src_rate: u32, dst_rate: u32) -> Vec<f32> {
    use std::f64::consts::PI;

    if input.is_empty() || src_rate == 0 || src_rate == dst_rate {
        return input.to_vec();
    }

    let lowpass_filter_width: usize = 6;
    let rolloff: f64 = 0.99;

    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = ((input.len() as f64) / ratio).ceil() as usize;

    let cutoff = rolloff * (dst_rate.min(src_rate) as f64) / (src_rate.max(dst_rate) as f64);

    let width = if src_rate > dst_rate {
        lowpass_filter_width as f64 * ratio
    } else {
        lowpass_filter_width as f64
    };

    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let center = i as f64 * ratio;
        let lo = (center - width).ceil().max(0.0) as usize;
        let hi = (center + width).floor().min((input.len() - 1) as f64) as usize;

        let mut sum = 0.0f64;
        let mut weight_sum = 0.0f64;

        for (j, &sample) in input[lo..=hi].iter().enumerate() {
            let x = (lo + j) as f64 - center;

            let sinc = if x.abs() < 1e-12 {
                cutoff
            } else {
                (PI * x * cutoff).sin() / (PI * x)
            };

            let t = x / width;
            let window = if t.abs() <= 1.0 {
                0.5 * (1.0 + (PI * t).cos())
            } else {
                0.0
            };

            let w = sinc * window;
            sum += sample as f64 * w;
            weight_sum += w;
        }

        let sample = if weight_sum.abs() > 1e-12 {
            sum / weight_sum
        } else {
            0.0
        };
        out.push(sample as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_sinc_identity_same_rate() {
        let input: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let output = resample_sinc(&input, 16000, 16000);
        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 1e-6, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn resample_sinc_empty_input() {
        let output = resample_sinc(&[], 44100, 16000);
        assert!(output.is_empty());
    }

    #[test]
    fn resample_sinc_downsample_correct_length() {
        let input: Vec<f32> = vec![0.0; 44100];
        let output = resample_sinc(&input, 44100, 16000);
        let expected = (44100.0f64 / (44100.0 / 16000.0)).ceil() as usize;
        assert_eq!(output.len(), expected);
    }
}
