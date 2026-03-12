/// Parameters for autoregressive generation.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub temperature: f64,
    pub top_p: f64,
    pub max_tokens: usize,
    pub eos_token_id: u32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 1.0,
            max_tokens: 700,
            eos_token_id: 7,
        }
    }
}

/// Sample a token from logits using temperature + optional top-p.
pub fn sample_logits(logits: &[f32], params: &GenerateParams) -> u32 {
    if params.temperature < 1e-10 {
        return argmax(logits);
    }

    // Apply temperature
    let scaled: Vec<f32> = logits
        .iter()
        .map(|&l| l / params.temperature as f32)
        .collect();

    if params.top_p < 1.0 {
        sample_top_p(&scaled, params.top_p)
    } else {
        sample_categorical(&softmax(&scaled))
    }
}

fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|e| e / sum).collect()
}

fn sample_top_p(scaled_logits: &[f32], top_p: f64) -> u32 {
    let probs = softmax(scaled_logits);

    let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0f32;
    let mut filtered = Vec::new();
    for (idx, prob) in &indexed {
        cumulative += prob;
        filtered.push((*idx, *prob));
        if cumulative >= top_p as f32 {
            break;
        }
    }

    // Renormalize
    let total: f32 = filtered.iter().map(|(_, p)| p).sum();
    let normalized: Vec<f32> = filtered.iter().map(|(_, p)| p / total).collect();

    let r = rand_f32();
    let mut cumulative = 0.0f32;
    for (i, prob) in normalized.iter().enumerate() {
        cumulative += prob;
        if r < cumulative {
            return filtered[i].0 as u32;
        }
    }

    filtered.last().map(|(idx, _)| *idx as u32).unwrap_or(0)
}

fn sample_categorical(probs: &[f32]) -> u32 {
    let r = rand_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Simple thread-local xorshift64 RNG.
fn rand_f32() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        );
    }
    STATE.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 40) as f32 / (1u64 << 24) as f32
    })
}

/// Parse a token string like `<|s_123|>` into the codec index.
pub fn parse_codec_token(token: &str) -> Option<u32> {
    let inner = token.strip_prefix("<|s_")?.strip_suffix("|>")?;
    inner.parse().ok()
}

/// Build the chat-formatted prompt for TTS generation.
pub fn build_prompt(text: &str) -> String {
    format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

/// Extract codec token indices from generated token IDs.
pub fn extract_codec_tokens(
    generated_ids: &[u32],
    id_to_token: &dyn Fn(u32) -> Option<String>,
) -> Vec<u32> {
    let mut codec_tokens = Vec::new();
    for &token_id in generated_ids {
        if let Some(token_str) = id_to_token(token_id) {
            if let Some(idx) = parse_codec_token(&token_str) {
                codec_tokens.push(idx);
            }
        }
    }
    codec_tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_codec_token_valid() {
        assert_eq!(parse_codec_token("<|s_0|>"), Some(0));
        assert_eq!(parse_codec_token("<|s_12799|>"), Some(12799));
        assert_eq!(parse_codec_token("<|s_42|>"), Some(42));
    }

    #[test]
    fn parse_codec_token_invalid() {
        assert_eq!(parse_codec_token("hello"), None);
        assert_eq!(parse_codec_token("<|s_|>"), None);
        assert_eq!(parse_codec_token("<|s_abc|>"), None);
        assert_eq!(parse_codec_token("<|im_start|>"), None);
    }

    #[test]
    fn build_prompt_format() {
        let prompt = build_prompt("こんにちは");
        assert_eq!(
            prompt,
            "<|im_start|>user\nこんにちは<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn sample_logits_greedy() {
        let logits = vec![0.0, 1.0, 0.5, -1.0];
        let params = GenerateParams {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(sample_logits(&logits, &params), 1);
    }

    #[test]
    fn sample_logits_returns_valid_index() {
        let logits = vec![0.1, 0.2, 0.3, 0.15, 0.25];
        let params = GenerateParams::default();
        for _ in 0..1000 {
            let idx = sample_logits(&logits, &params);
            assert!((idx as usize) < logits.len());
        }
    }

    #[test]
    fn rand_f32_in_unit_range() {
        for _ in 0..10000 {
            let r = rand_f32();
            assert!(r >= 0.0, "rand_f32 returned {r} < 0");
            assert!(r < 1.0, "rand_f32 returned {r} >= 1");
        }
    }
}
