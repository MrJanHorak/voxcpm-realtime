# VoxCPM Real-Time Voice Engine

Independent local VoxCPM2 LoRA voice validation and persistent inference engine. The current native backend is experimental and has not achieved real-time speed on the tested GPU. The existing `voiceover-automation` project is **not modified**.

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
- Native local inference and benchmarking are implemented; experimental audio.cpp streaming is available. Watermarking and full client SDKs are not implemented.
- The validation API accepts local filesystem paths and is intentionally localhost-only; add path restrictions and authentication before remote deployment.
- Keep private voice recordings and LoRA weights out of version control. Only use voices with permission and retain required model watermarking and AI-generated-audio disclosures.

## Phase 2 prerequisites

1. Inspect the actual `lora_config.json` and `voice.json` for model identifiers and LoRA keys.
2. Pin the exact VoxCPM2 base checkpoint and Python/runtime versions.
3. Implement one-shot generation with the native `voxcpm` runtime and measure peak GPU memory on the RTX 3050 4 GB.
4. Evaluate the persistent `audio.cpp` runtime separately for Q8 GPU deployment, with a verified LoRA merge/loading path.
5. Only after successful audio generation, implement chunked streaming and latency tests.

## Phase 2: native VoxCPM2 first WAV (experimental)

See [Phase 2 setup](docs/PHASE2.md). Install with `pip install -e ".[dev,inference]"` after configuring a compatible CUDA PyTorch build. Run `voxcpm-speak --model-dir ./models/VoxCPM2 --voice-dir ./voices/step150 --text "Hello, this is a test." --output ./outputs/first.wav`.

This is not streaming. Base-model and adapter assets are not committed. See [Phase 3](docs/PHASE3.md) for the persistent server and measured results.

## Phase 3: persistent local inference

Set `VOXCPM_MODEL_DIR`, `VOXCPM_VOICE_DIR`, and optionally `VOXCPM_REFERENCE_WAV` before starting uvicorn. Use **one worker** on localhost. `POST /api/warmup` loads the model and generates a warmup; `POST /api/synthesize` accepts `{"text":"Hello.","seed":35}` and returns WAV audio. Requests serialize within the process. See [Phase 3](docs/PHASE3.md) for setup, measurements, and limitations.

## Phase 4: experimental Q8 backend

The locally evaluated audio.cpp backend is substantially faster on the tested RTX 3050. See [Phase 4](docs/PHASE4.md) for the measured results, limitations, and separate local server instructions. The user reported that the test samples sounded good; watermark presence remains unverified.

## Phase 5: experimental streaming

Start the audio.cpp server in streaming mode and receive or play audio as it is generated. See [Phase 5](docs/PHASE5.md) for commands, API details, measured first-audio latency, and limitations.
