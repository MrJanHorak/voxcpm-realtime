# VoxCPM Real-Time Voice Engine — Phase 1

Independent Python scaffold for checking VoxCPM2 LoRA voice packages before building low-latency inference. The existing `voiceover-automation` project is **not modified**.

## Source compatibility

The training project's documentation identifies VoxCPM2 (`voxcpm` 2.0.3), a LoRA checkpoint directory containing `lora_config.json` and `lora_weights.safetensors`, and a specific base-model revision. Do not assume adapters are compatible with other revisions. See https://github.com/MrJanHorak/voiceover-automation/blob/main/training/TRAINED_VOICES.md

## Setup (Windows PowerShell, Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
```

Copy a **private** checkpoint directory to `voices/step150/`, preserving the two original filenames. Optionally add `reference.wav`, `transcript.txt` and `voice.json`. No voice assets are bundled.

```powershell
voicecheck .\voices\step150
python -m uvicorn voxcpm_realtime.api:app --host 127.0.0.1 --port 8765
```

Then visit http://127.0.0.1:8765/docs or request `GET /health` and `GET /api/capabilities`. `POST /api/voices/validate` accepts `{"folder":"voices/step150"}`.

## Scope and limitations

- Validates JSON, Safetensors readability/tensor names and optional WAV metadata; no arbitrary model code is executed.
- `compatible: true` means **static checks passed**, NOT that the adapter matches the installed base model or generates correct speech.
- Inference, model download, LoRA loading, CUDA benchmarks, streaming, audio watermarking and client SDKs are **not yet implemented**.
- The validation API accepts local filesystem paths and is intentionally localhost-only; add path restrictions and authentication before remote deployment.
- Keep private voice recordings and LoRA weights out of version control. Only use voices with permission and retain required model watermarking and AI-generated-audio disclosures.

## Phase 2 prerequisites

1. Inspect the actual `lora_config.json` and `voice.json` for model identifiers and LoRA keys.
2. Pin the exact VoxCPM2 base checkpoint and Python/runtime versions.
3. Implement one-shot generation with the native `voxcpm` runtime and measure peak GPU memory on the RTX 3050 4 GB.
4. Evaluate the persistent `audio.cpp` runtime separately for Q8 GPU deployment, with a verified LoRA merge/loading path.
5. Only after successful audio generation, implement chunked streaming and latency tests.
