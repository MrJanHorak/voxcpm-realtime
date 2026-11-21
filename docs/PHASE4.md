> **Phase 6 correction:** The existing step150 metadata requires a Perth implicit watermark. The raw output described here does not apply it. Use the buffered [watermarked client](PHASE6.md) for this voice.

# Phase 4 — audio.cpp Q8 evaluation

This is an experimental, separate local server. It does not replace the Phase 3 FastAPI server. The test used audio.cpp v0.9.5, commit `61c7838baf1f7f34769b5b7bd369c0d78edf9535`, the official Windows CUDA 12.4 release and runtime archives. Their SHA-256 digests were checked against the GitHub release metadata. The current checkout and archives are under ignored `outputs/`; they are not shipped in Git. The exact local VoxCPM2 revision `7936080b41163874f38610f46f438e2c0757f9fd` was confirmed in its Hugging Face download metadata. The local checkpoint and adapter were never uploaded.

## Start the tested server

The files prepared during the evaluation must still be present: `outputs/audio-cpp-bin/`, `outputs/audio-cpp-runtime/`, and `outputs/voxcpm2-q8-local.gguf`. Stop the native server first to free GPU memory. From the project directory:

```powershell
.\scripts\start_audio_cpp.ps1
```

In a second PowerShell window:

```powershell
Invoke-WebRequest -Uri 'http://127.0.0.1:8766/v1/audio/speech' `
  -Method Post -ContentType 'application/json' `
  -Body '{"model":"step150","input":"Hello, this is a test of the voice engine.","seed":35}' `
  -OutFile 'outputs/q8-test.wav'
Start-Process 'outputs/q8-test.wav'
```

The first request loads the model and adapter; later requests reuse them and cache the reference. The script accepts `-Executable`, `-RuntimeDir`, `-Model`, `-VoiceDir`, `-Reference`, and `-Port` to select other local paths. Paths default to this project and the `step150` voice. This server listens on localhost and has a different API from Phase 3. Treat it as local-only. Do not run both GPU servers at once on the 4 GiB card. The original server and `voxcpm-speak` remain available.

## Conversion used for this test

The source checkout is `outputs/audio-cpp` at the release commit. First convert `models/VoxCPM2/audiovae.pth` to a temporary safetensors file with audio.cpp's `tests/voxcpm2/convert_pth_to_safetensors.py`; this checks all converted tensor values against the source. Put only `config.json`, `tokenizer_config.json`, `tokenizer.json`, and `special_tokens_map.json` from the local base in `outputs/voxcpm2-sidecars`. The exact conversion command was:

```powershell
.\outputs\audio-cpp-bin\audiocpp_gguf.exe `
  --input weights=models/VoxCPM2/model.safetensors `
  --input audiovae_weights=outputs/audiovae.safetensors `
  --root outputs/voxcpm2-sidecars --output outputs/voxcpm2-q8-local.gguf `
  --type q8_0 --family voxcpm2 --model-spec outputs/audio-cpp/model_specs/voxcpm2.json `
  --keep-type 'audiovae_weights/*=f16' `
  --exclude-prefix 'audiovae_weights/decoder.sr_bin_boundaries'
```

The excluded integer tensor is unsupported by this converter; the audio.cpp VoxCPM2 implementation reads sample-rate boundaries from the model configuration. The AudioVAE is kept F16 while the model uses the converter's Q8 rules. The resulting file is 2,766,633,504 bytes. The adapter remains separate and is merged in memory at session startup; its source files are unchanged. This conversion is tied to the tested audio.cpp release, model and configuration; it is not validated as a general converter for all models.

## Measurements, 2026-11-21

RTX 3050 Laptop 4 GiB, CUDA backend, local `step150` adapter and reference, seed 35, text `Hello, this is a test of the voice engine.` Three requests in one CLI session produced 2.56 s each at 48 kHz: 1.731 s (RTF 0.676), then 1.022 s (RTF 0.399), then 1.023 s (RTF 0.400). The latter two reused the encoded reference and embedding. These metrics exclude model/session loading; output length and randomness differ from the native backend, so this is not a strict numerical or quality comparison. The Phase 3 native test took 57.80 and 57.71 s for 2.88 s audio with the same text and seed.

A separate HTTP test took 13.27 s for the cold first request, including loading and adapter merging, and 1.07 s for the second; both returned 48 kHz WAV. The tested startup script also returned HTTP 200 and a valid WAV. audio.cpp logged 192 merged LoRA matrices, matching 384 A/B adapter tensors; this is not proof of speaker identity or perceptual quality. The backend reported model and AudioVAE buffers of approximately 2,456 and 358 MiB, respectively. A single device-wide `nvidia-smi` observation during inference was 3,827 MiB of 4,096 MiB; it is not a sampled peak and includes other processes. Shared GPU memory was not measured in this test.

The runtime reported that libsoxr was unavailable and used linear reference-audio resampling; the effect on voice quality is unmeasured. Output was non-silent. The user listened to the generated samples and reported that they sounded good; no systematic intelligibility or speaker-similarity evaluation was performed, and watermark detection was not tested. As in Phase 3, no audio watermarking step is known to be present; do not assume outputs are watermarked. Do not interpret these few timings as a guarantee for longer texts or other workloads. Streaming, concurrent requests and the automatic shutdown behavior under load were not tested.
