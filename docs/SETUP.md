# Dedicated Windows runtime installation

The installer prepares the tested audio.cpp v0.9.5 Windows x64 CUDA 12.4 runtime and converts the exact local VoxCPM2 base checkpoint into Q8 GGUF. The CUDA build targets NVIDIA GPUs and requires a compatible driver. It does not download, copy, or upload voice packages, reference recordings, or adapter weights.

## Prerequisites

- Windows 11 x64, NVIDIA CUDA-capable GPU and compatible driver.
- An existing Python environment configured for this project, including CPU-capable PyTorch, `safetensors`, and the existing inference and watermark dependencies. The validated machine uses Python 3.13, PyTorch 2.8 CUDA, and VoxCPM 2.0.3. PyTorch's CPU tensors are used for conversion; the existing GPU PyTorch install is not replaced by the installer.
- The **exact** base-model revision `7936080b41163874f38610f46f438e2c0757f9fd` in `models/VoxCPM2`, including its AudioVAE and tokenizer/config files. Obtain it using the pinned download example in [Phase 2](PHASE2.md). The installer verifies SHA-256 for the model, AudioVAE and four sidecars before conversion; it rejects mismatches.
- Enough local disk space for the archives, extracted runtime, temporary conversion data, and final GGUF. The GPU is not needed for conversion. A compatible Python environment can be prepared with `pip install -e ".[dev,inference,watermark,playback]"` after installing an appropriate PyTorch build. The watermark package requires the PyTorch and audio-processing dependencies already supplied by the working inference environment.

From the project directory with `.venv` active:

```powershell
python -m voxcpm_realtime.install_runtime
```

The installer looks in `outputs/` for the previously downloaded release archives. If missing, it downloads the pinned archives directly from the official [audio.cpp v0.9.5 release](https://github.com/0xShug0/audio.cpp/releases/tag/v0.9.5). The binary and CUDA runtime downloads total about 1.08 GB. `--archives <directory>` chooses another source directory; if the required archives are absent there, the installer downloads them. Existing but corrupt archives cause an error. Both archives are checked against pinned SHA-256 values before extraction; no private voice data is sent.

The dedicated paths are:

- `.runtime/audio-cpp-v0.9.5/bin/` — release executables and model specifications.
- `.runtime/audio-cpp-v0.9.5/runtime/` — matching CUDA runtime DLLs.
- `models/VoxCPM2-GGUF/voxcpm2-q8_0.gguf` — locally converted base model.
- `models/VoxCPM2-GGUF/provenance.json` — expected base hashes, revision, conversion recipe summary and generated GGUF hash.

The manager and PowerShell launcher prefer this complete installation. If it is absent, they fall back to the older `outputs/` paths for compatibility. Generated WAVs, logs and transient server configuration can still be written under `outputs/`, but the manager and launcher recreate that directory if needed; their runtime and model no longer depend on it. `.runtime/`, `models/`, `voices/`, and `outputs/` remain ignored by Git.

The installer stages files in temporary directories inside the project and publishes them only after successful extraction or conversion. It refuses to replace an existing installation or model. On a repeat run it checks recorded critical-file and GGUF hashes and reuses valid files; if existing files are incomplete or changed it stops with an error. These local manifest checks detect changes against the recorded installation; they are not protection against someone who can alter both files and manifests. The AudioVAE conversion uses `torch.load(..., weights_only=True)` on the checksum-verified file, verifies its resulting tensors, and invokes the pinned audio.cpp converter. The model conversion retains the Phase 4 Q8/F16 recipe and excludes the integer `decoder.sr_bin_boundaries` tensor, which the audio.cpp implementation obtains from configuration. The generated GGUF's byte hash need not match an independently converted file.

## Use and verification

Use the [voice manager](PHASE6_MANAGER.md), the [watermarked offline client](PHASE6.md), or the [watermarked streaming client](PHASE6_STREAMING.md) as before. The raw audio.cpp server and manager relay do not apply the required watermark; use the clients. Watermarked streaming remains experimental, and its short-excerpt and failure limitations still apply.

On 2026-11-21, installation on the RTX 3050 Laptop reused the existing downloaded archives, verified all six local base files and converted the local model. A second installer run verified and reused the installed files. The manager and standalone launcher both selected the dedicated model and CUDA DLL paths. With `step150`, a watermarked streaming test generated 3.52 seconds in three verified segments; the saved WAV scored 0.946 with the local Perth detector. An offline watermarked test generated 3.52 seconds and scored 1.0. These are smoke tests, not a watermark robustness or listening-quality assessment. The newly converted GGUF has a different byte hash from the earlier conversion; both were made from the verified base checkpoint, but byte-for-byte equivalence has not been established. The official download fallback was not exercised in this installation because the verified archives were already present.
