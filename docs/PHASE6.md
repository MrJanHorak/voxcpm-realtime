# Phase 6 — Private voice packages and watermarked output

## Important watermark correction

The existing `voices/step150/voice.json` explicitly says to apply the Perth implicit watermark. This was discovered during Phase 6: the Phase 2–5 generated samples were not passed through a known watermarking step. The raw audio.cpp offline and streaming endpoints, the Phase 3 FastAPI endpoint, `voxcpm-speak`, and the low-level `stream_speech()` function do **not** apply the required watermark. Do not treat their output as watermarked. The server launcher displays a warning. Phase 6 adds a separate buffered client that applies and checks Perth before returning audio. The Python streaming client now has an [experimental watermarked mode](PHASE6_STREAMING.md).

## Package format

A package is a private directory under `voices/<id>/`, with `lora_config.json`, `lora_weights.safetensors`, optional `reference.wav`, and a `voxcpm-voice.json` manifest. The manifest records schema version 1, directory-matching ID, base-model revision, reference filename or null, and `perth-implicit` watermark requirement. The existing `voice.json` is retained unchanged. Packages, recordings and weights remain excluded by `.gitignore`.

`step150` was initialized locally from its existing metadata without copying or altering the original voice files. To inspect or initialize a compatible voice whose existing `voice.json` declares the supported base revision and Perth implicit watermark:

```powershell
python -m voxcpm_realtime.voices list
python -m voxcpm_realtime.voices check voices/step150
# For another compatible folder with existing voice.json metadata:
python -m voxcpm_realtime.voices init voices/<id>
```

Initialization refuses to overwrite an existing package manifest. It performs static validation only. It checks the manifest's declared revision against the supported revision; it does not prove the selected GGUF was derived from that revision, the adapter fits it, or that a recording is appropriate. Only `step150` has been tested. The [local voice manager](PHASE6_MANAGER.md) can switch packages by restarting its owned backend while retaining one loaded voice. Simultaneous voices are not supported.

## Watermarked offline synthesis

Install `pip install -e ".[watermark]"` into the project's virtual environment; `resemble-perth==1.0.1` was installed for the local test. The existing inference setup supplies PyTorch and the audio processing dependencies. With the [dedicated runtime](SETUP.md) installed (or the older Phase 4 files in place), stop other GPU servers, then:

```powershell
.\scripts\start_audio_cpp.ps1 -VoiceId step150
```

In a second window with `.venv` active:

```powershell
python -m voxcpm_realtime.client --model step150 `
  --text 'Hello, this is a watermarked voice test.' `
  --output outputs/watermarked-test.wav
Start-Process outputs/watermarked-test.wav
```

The client accepts only local HTTP addresses. It requests the full WAV from the offline server, applies Perth on CPU, encodes 48 kHz mono PCM16 WAV, then checks the encoded result with the Perth detector. A detector score below 0.5, a watermark error, or invalid audio causes the client to return no result. This is a local detector check, not a guarantee against removal or all transformations. Output is buffered rather than streamed. Existing files are not overwritten. Reusing one `VoiceClient` instance caches the CPU watermarker.

```python
from voxcpm_realtime.client import VoiceClient
client = VoiceClient(model='step150')
watermarked_wav_bytes = client.synthesize('Hello, this is a test.')
```

For another validated package, start with `-VoiceId <id>` and pass the same `--model <id>` or `VoiceClient(model='<id>')`. The launcher still supports the original `-VoiceDir` and `-Reference` arguments for compatibility; those and the raw backend are not a watermark enforcement boundary. The backend model and private reference stay local.

## Local test, 2026-11-21

The package-selected `step150` server generated a 2.56-second 48 kHz WAV through the new client. The final saved WAV scored 1.0 with the installed Perth implicit detector. On a separate existing 2.56-second sample, a CPU application took about 0.034 seconds after the watermarker was initialized; the unmarked sample scored approximately 0.00036 and the marked sample 1.0. These are individual tests, not a robustness or perceptual-quality study. The watermarked output was not independently listened to. No second trained voice, voice switching under load, concurrent clients, long-form synthesis, or watermark-preserving live streaming was tested at that time.
