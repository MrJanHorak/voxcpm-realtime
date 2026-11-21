# Phase 3 — Persistent local inference

The server keeps one base model and one LoRA adapter in memory per process. Set the three environment variables below before starting it; the reference is optional. Relative paths are resolved against the server's working directory.

```powershell
$env:VOXCPM_MODEL_DIR = 'models/VoxCPM2'
$env:VOXCPM_VOICE_DIR = 'voices/step150'
$env:VOXCPM_REFERENCE_WAV = 'voices/step150/reference.wav'
python -m uvicorn voxcpm_realtime.api:app --host 127.0.0.1 --port 8765 --workers 1
Invoke-RestMethod -Method Post http://127.0.0.1:8765/api/warmup
Invoke-WebRequest -Method Post http://127.0.0.1:8765/api/synthesize -ContentType 'application/json' -Body '{"text":"Hello, this is a test.","seed":35}' -OutFile outputs/api.wav
```

The model loads lazily, on the first warmup or synthesis request. Warmup performs a complete generation and may take substantial time. Requests wait on a process-local lock; use exactly one worker and one server process per GPU. The API has no authentication and exposes a filesystem validation endpoint: keep it on localhost. Errors during model or CUDA operation return server errors; check server logs. The Phase 2 `voxcpm-speak` command remains available as a separate one-shot path.

The installed VoxCPM 2.0.3 `VoxCPM2Model.build_prompt_cache` encodes reference audio and `generate_with_prompt_cache` consumes the result. The manager caches one reference, rebuilding it if file size or modification time changes. It uses the same generation defaults and whitespace normalization as the high-level `VoxCPM.generate`, including bad-case retries; it disables denoising and compilation, as in Phase 2. The public wrapper rebuilds the prompt cache per call. Model and adapter are fixed for the lifetime of the process; restart after replacing them. Adapter loading rejects zero loaded or any skipped checkpoint keys. This does not prove that every model LoRA parameter was provided or verify the model revision.

Neither the installed VoxCPM inference code nor the previous one-shot implementation contains an identifiable audio watermarking step. This change does not remove a watermarking step, but watermark presence and any external watermarking requirement are **not verified**. Do not assume the returned WAV contains a watermark. The [VoxCPM documentation](https://voxcpm.readthedocs.io/en/latest/) recommends labeling publicly shared generated audio as AI-generated.

Run the benchmark with:

```powershell
python -m voxcpm_realtime.benchmark --model-dir models/VoxCPM2 --voice-dir voices/step150 --reference-wav voices/step150/reference.wav --runs 2 --output outputs/benchmark.wav
```

The benchmark reports model and adapter load time, first warmup time including loading and reference encoding, subsequent full generation plus WAV write time, output duration and real-time factor (elapsed/audio duration). PyTorch peaks measure allocator bytes, not dedicated residency. `nvidia-smi` samples device-wide dedicated memory including other applications; it can miss peaks. On Windows, process shared GPU memory is sampled from WDDM counters across adapter instances; `null` means unavailable. These sampled measurements are approximate and should not be added to PyTorch's allocator values. All generated WAVs and local model/voice assets are ignored by Git.

## Measured on the target laptop, 2026-11-21

With the local base model and `step150` adapter, reference WAV, VoxCPM 2.0.3, PyTorch 2.8 CUDA, bfloat16 and RTX 3050 Laptop 4 GiB, a fresh process measured 12.92 s for base-model and LoRA loading and 69.57 s for first warmup including load and reference encoding. Two warmed runs of the same sentence with seed 35 took 57.80 s and 57.71 s including WAV writing; each produced 2.88 s of 48 kHz audio, for real-time factors 20.07 and 20.04. These results are one trial under the then-current system load, not a quality or latency guarantee.

PyTorch reported peaks of 6,002,440,192 allocated bytes and 6,530,531,328 reserved bytes; these are allocator totals, not dedicated VRAM residency. Sampled device-wide dedicated usage reached 3,940 of 4,096 MiB. A separate Windows process counter observation during that run showed 2,640,314,368 bytes of shared GPU memory for the inference Python process. The benchmark at the time did not yet collect shared memory automatically; the later sampler reports it where Windows counters are available. Together the figures strongly support oversubscription and shared-memory use as a likely contributor to latency, but no controlled comparison establishes its precise contribution. GPU workload, model compute and other system activity may also contribute. No speedup relative to an identical one-shot workload was measured.

A separate in-process FastAPI `TestClient` smoke test loaded the real model and reference and returned HTTP 200 with a valid 48 kHz, 0.8-second WAV for `Hello.`. The updated sampler measured a 2,675,965,952-byte process shared-memory peak and 3,940 MiB device-wide dedicated peak in that run. It did not test network transport, concurrent real GPU requests, subjective voice quality, or watermark detection. Unit tests cover serialization and cache invalidation with fakes.
