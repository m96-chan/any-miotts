//! Synthesize speech from text using any-miotts.
//!
//! Usage:
//!   cargo run -p any-miotts --features candle-cuda --example synthesize -- "こんにちは" --out /tmp/test.wav
//!   cargo run -p any-miotts --example synthesize -- "Hello world" --wav ref.wav --out output.wav
//!   cargo run -p any-miotts --example synthesize -- "First. Second. Third." --stream

use std::path::PathBuf;
use std::time::Instant;

use any_miotts::{SynthesisEvent, TtsError};

#[tokio::main]
async fn main() -> Result<(), TtsError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let parsed = parse_args(&args);

    tracing::info!("Text: {}", parsed.text);
    tracing::info!("Reference WAV: {}", parsed.wav.display());
    tracing::info!("Streaming: {}", parsed.stream);

    let t0 = Instant::now();
    let engine = any_miotts::initialize(&parsed.wav).await?;
    let sample_rate = engine.sample_rate();
    tracing::info!("Engine initialized in {:.1}s", t0.elapsed().as_secs_f32());

    tracing::info!("Warmup...");
    let t_warm = Instant::now();
    engine.warmup().ok();
    tracing::info!(
        "Warmup done in {:.1}ms",
        t_warm.elapsed().as_secs_f64() * 1000.0
    );

    if parsed.stream {
        synthesize_streaming(&engine, &parsed.text, sample_rate as u32, parsed.out.as_deref())?;
    } else {
        synthesize_batch(&engine, &parsed.text, sample_rate as u32, parsed.out.as_deref())?;
    }

    Ok(())
}

fn synthesize_batch(
    engine: &any_miotts::TtsEngine,
    text: &str,
    sample_rate: u32,
    out_path: Option<&std::path::Path>,
) -> Result<(), TtsError> {
    let t1 = Instant::now();
    let pcm = engine.synthesize(text)?;
    let dur = t1.elapsed();
    let audio_secs = pcm.len() as f32 / sample_rate as f32;
    tracing::info!(
        "Synthesized {:.2}s audio in {:.2}s (RTF={:.3})",
        audio_secs,
        dur.as_secs_f32(),
        dur.as_secs_f32() / audio_secs,
    );

    if let Some(path) = out_path {
        write_wav(path, &pcm, sample_rate);
        tracing::info!("Wrote {}", path.display());
    } else {
        tracing::info!("Playing audio...");
        play_pcm(&pcm, sample_rate);
    }

    Ok(())
}

fn synthesize_streaming(
    engine: &any_miotts::TtsEngine,
    text: &str,
    sample_rate: u32,
    out_path: Option<&std::path::Path>,
) -> Result<(), TtsError> {
    let t1 = Instant::now();
    let rx = engine.synthesize_streaming(text);

    let mut all_pcm = Vec::new();
    let mut sentence_idx = 0usize;

    for event in rx {
        match event? {
            SynthesisEvent::SentenceAudio { text, pcm } => {
                let audio_secs = pcm.len() as f32 / sample_rate as f32;
                tracing::info!(
                    "[sentence {}] {:.2}s audio for: {:?}",
                    sentence_idx,
                    audio_secs,
                    text
                );

                if out_path.is_none() {
                    // Play each chunk immediately for low-latency streaming
                    tracing::info!("Playing sentence {}...", sentence_idx);
                    play_pcm(&pcm, sample_rate);
                }

                all_pcm.extend_from_slice(&pcm);
                sentence_idx += 1;
            }
            SynthesisEvent::Done => {
                tracing::info!("Streaming synthesis complete.");
            }
        }
    }

    let dur = t1.elapsed();
    let total_audio_secs = all_pcm.len() as f32 / sample_rate as f32;
    tracing::info!(
        "Total: {:.2}s audio in {:.2}s (RTF={:.3})",
        total_audio_secs,
        dur.as_secs_f32(),
        if total_audio_secs > 0.0 {
            dur.as_secs_f32() / total_audio_secs
        } else {
            0.0
        },
    );

    if let Some(path) = out_path {
        write_wav(path, &all_pcm, sample_rate);
        tracing::info!("Wrote {}", path.display());
    }

    Ok(())
}

fn write_wav(path: &std::path::Path, pcm: &[f32], sample_rate: u32) {
    let peak = pcm.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let gain = if peak > 1e-6 { 0.9 / peak } else { 1.0 };
    let samples_i16: Vec<i16> = pcm
        .iter()
        .map(|&s| (s * gain * 32767.0).clamp(-32768.0, 32767.0) as i16)
        .collect();
    let data_bytes: Vec<u8> = samples_i16
        .iter()
        .flat_map(|s| s.to_le_bytes())
        .collect();
    let data_len = data_bytes.len() as u32;
    let file_len = 36 + data_len;
    let byte_rate: u32 = sample_rate * 2;
    let mut wav_bytes = Vec::with_capacity(44 + data_bytes.len());
    wav_bytes.extend_from_slice(b"RIFF");
    wav_bytes.extend_from_slice(&file_len.to_le_bytes());
    wav_bytes.extend_from_slice(b"WAVE");
    wav_bytes.extend_from_slice(b"fmt ");
    wav_bytes.extend_from_slice(&16u32.to_le_bytes());
    wav_bytes.extend_from_slice(&1u16.to_le_bytes());
    wav_bytes.extend_from_slice(&1u16.to_le_bytes());
    wav_bytes.extend_from_slice(&sample_rate.to_le_bytes());
    wav_bytes.extend_from_slice(&byte_rate.to_le_bytes());
    wav_bytes.extend_from_slice(&2u16.to_le_bytes());
    wav_bytes.extend_from_slice(&16u16.to_le_bytes());
    wav_bytes.extend_from_slice(b"data");
    wav_bytes.extend_from_slice(&data_len.to_le_bytes());
    wav_bytes.extend_from_slice(&data_bytes);
    std::fs::write(path, &wav_bytes).expect("Failed to write WAV");
}

fn play_pcm(pcm: &[f32], sample_rate: u32) {
    use rodio::{OutputStream, Sink};

    let (_stream, handle) = OutputStream::try_default().expect("No audio output device");
    let sink = Sink::try_new(&handle).expect("Sink creation failed");

    let source = rodio::buffer::SamplesBuffer::new(1, sample_rate, pcm.to_vec());
    sink.append(source);
    sink.sleep_until_end();
}

struct ParsedArgs {
    text: String,
    wav: PathBuf,
    out: Option<PathBuf>,
    stream: bool,
}

fn parse_args(args: &[String]) -> ParsedArgs {
    let mut text = None;
    let mut wav = PathBuf::from("reference.wav");
    let mut out = None;
    let mut stream = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--wav" => {
                i += 1;
                wav = PathBuf::from(&args[i]);
            }
            "--out" => {
                i += 1;
                out = Some(PathBuf::from(&args[i]));
            }
            "--stream" => {
                stream = true;
            }
            "--help" | "-h" => {
                eprintln!(
                    "Usage: synthesize <TEXT> [--wav <ref.wav>] [--out <output.wav>] [--stream]"
                );
                std::process::exit(0);
            }
            _ => {
                if text.is_none() {
                    text = Some(args[i].clone());
                }
            }
        }
        i += 1;
    }

    let text = text.unwrap_or_else(|| {
        eprintln!("Usage: synthesize <TEXT> [--wav <ref.wav>] [--out <output.wav>] [--stream]");
        std::process::exit(1);
    });

    ParsedArgs {
        text,
        wav,
        out,
        stream,
    }
}
