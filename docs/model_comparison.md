# MioTTS Model Size Comparison

Snapdragon 8 Elite (ARM, Q4_0, CPU dotprod+i8mm, 4 threads) での比較。

## Performance

| Model | Size | Android tok/s | RTF | Realtime? |
|-------|------|---------------|-----|-----------|
| 2.6B Q4_0 | 1.4 GB | 12.1 | 2.0 | No |
| 1.2B Q4_0 | 683 MB | 23.9 | 1.05 | Borderline |
| 0.6B Q4_0 | 375 MB | 27.6 | 0.91 | Yes |
| 0.1B Q4_0 | 75 MB | 136.6 | 0.18 | Yes |

- RTF (Real-Time Factor): < 1.0 = faster than realtime
- Codec rate: 25 Hz (25 tokens = 1 second of audio)
- Device: Snapdragon 8 Elite, Oryon CPU, LPDDR5X

## Audio Samples

`tts_model_comparison.zip` contains 16 WAV files (4 models x 4 test sentences).

### Test Sentences

| ID | Language | Text | Purpose |
|----|----------|------|---------|
| test0 | JP | こんにちは、今日はいい天気ですね。 | Short, simple |
| test1 | JP | おはようございます。今日の予定を確認しましょう。午前中は会議があります。午後からは自由時間です。 | Long, multi-sentence |
| test2 | JP | えっ、本当に？信じられない！すごいじゃん！ | Emotional, expressive |
| test3 | EN | The quick brown fox jumps over the lazy dog. This is a test of English speech synthesis. | English |

### File Naming

`{model}_test{id}.wav` — e.g. `0.6B-Q4_0_test2.wav`

## Quality Notes

- **2.6B**: Highest quality, natural prosody and emotion
- **1.2B**: Slightly faster tempo, otherwise comparable to 2.6B
- **0.6B**: Good quality, suitable for production use on mobile
- **0.1B**: Noticeable quality drop but acceptable for edge/IoT devices

## Recommendation

| Use Case | Model | Rationale |
|----------|-------|-----------|
| Desktop / Server | 2.6B | Best quality, GPU handles speed |
| Mobile (flagship) | 0.6B - 1.2B | Realtime on Snapdragon 8 Elite |
| Edge / IoT | 0.1B | 75 MB, runs anywhere |

## Reference Speaker

All samples use the same reference WAV for voice cloning (zero-shot).

## Test Environment

- Host: RTX 5090 + Ryzen 9950X3D (for MioCodec decode, candle-cuda)
- Target: Snapdragon 8 Elite (LFM2 inference, llama.cpp CPU)
- Quantization: Q4_0 (all models, from BF16 source via llama-quantize)
- ARM extensions: dotprod + i8mm enabled
- Thread count: 4 (auto-detected via sysfs cpufreq)
