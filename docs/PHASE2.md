# Phase 2 — First VoxCPM2 WAV

The project's existing training scripts in [voiceover-automation](https://github.com/MrJanHorak/voiceover-automation/blob/main/training/make_samples.py) are the source of the native inference call pattern.

## Prerequisites

- Python 3.11, VoxCPM 2.0.3, a compatible PyTorch/CUDA installation, and the matching VoxCPM2 base-model files.
- Base model revision **7936080b41163874f38610f46f438e2c0757f9fd**, matching the training script.
- A private LoRA checkpoint folder with `lora_config.json` and `lora_weights.safetensors`.
- Optional mono reference WAV.

Install the scaffold: `pip install -e ".[dev,inference]"`. GPU-enabled PyTorch installation varies by CUDA/driver and must be checked separately; the inference extra does **not** pin or install torch.

Download the base model to a local folder using the Hugging Face CLI, for example:

```powershell
python -m pip install "huggingface_hub[cli]"
hf download openbmb/VoxCPM2 --revision 7936080b41163874f38610f46f438e2c0757f9fd --local-dir .\models\VoxCPM2
```

Then:

```powershell
voxcpm-speak --model-dir .\models\VoxCPM2 --voice-dir .\voices\step150 --reference-wav .\voices\step150\reference.wav --text "Hello, this is a test." --output .\outputs\first.wav
```

Omit `--reference-wav` to test adapter-only speech. Start with a short sentence. Monitor VRAM using `nvidia-smi`; the 4 GB RTX 3050 may run out of memory with the native PyTorch model. Do not assume it fits or use unverified optimization flags. If native inference fails due to memory, next evaluate the audio.cpp GGUF path already documented by the training project.

## Current limitations

This is **one-shot**, **process-per-request** synthesis, not low-latency or real-time. It does not yet expose inference over the API or guarantee the model's watermarking behavior. The audio path, version compatibility and GPU requirements have **not been exercised on the target RTX 3050**. Keep generated output clearly labeled AI-generated and retain any required watermark.
