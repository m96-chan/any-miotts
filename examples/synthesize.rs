//! Synthesize speech from text using any-miotts.
//!
//! Usage:
//!   cargo run -p any-miotts --features candle-cuda --example synthesize -- "こんにちは" --out /tmp/test.wav
//!   cargo run -p any-miotts --example synthesize -- "Hello world" --wav ref.wav --out output.wav

use std::path::PathBuf;
use std::time::Instant;

use any_miotts::TtsError;

#[tokio::main]
async fn main() -> Result<(), TtsError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().collect();
    let (text, wav_path, out_path) = parse_args(&args);

    tracing::info!("Text: {text}");
    tracing::info!("Reference WAV: {}", wav_path.display());

    let t0 = Instant::now();
    let engine = any_miotts::initialize(&wav_path).await?;
    let sample_rate = engine.sample_rate();
    tracing::info!("Engine initialized in {:.1}s", t0.elapsed().as_secs_f32());

    tracing::info!("Warmup...");
    let t_warm = Instant::now();
    engine.warmup().ok();
    tracing::info!(
        "Warmup done in {:.1}ms",
        t_warm.elapsed().as_secs_f64() * 1000.0
    );

    let t1 = Instant::now();
    let pcm = engine.synthesize(&text)?;
    let dur = t1.elapsed();
    let audio_secs = pcm.len() as f32 / sample_rate as f32;
    tracing::info!(
        "Synthesized {:.2}s audio in {:.2}s (RTF={:.3})",
        audio_secs,
        dur.as_secs_f32(),
        dur.as_secs_f32() / audio_secs,
    );

    if let Some(path) = out_path {
        write_wav(&path, &pcm, sample_rate as u32);
        tracing::info!("Wrote {}", path.display());
    } else {
        tracing::info!("Playing audio...");
        play_pcm(&pcm, sample_rate as u32);
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

fn parse_args(args: &[String]) -> (String, PathBuf, Option<PathBuf>) {
    let mut text = None;
    let mut wav = PathBuf::from("reference.wav");
    let mut out = None;

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
            "--help" | "-h" => {
                eprintln!(
                    "Usage: synthesize <TEXT> [--wav <ref.wav>] [--out <output.wav>]"
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
        eprintln!("Usage: synthesize <TEXT> [--wav <ref.wav>] [--out <output.wav>]");
        std::process::exit(1);
    });

    (text, wav, out)
}
