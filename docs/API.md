# Local API and browser test page

The local manager serves a browser page and a buffered, watermark-enforcing HTTP endpoint. It runs on this computer and must not be exposed directly to untrusted networks: it has no authentication or TLS, and no cross-origin access policy is provided. Other browser origins need their own appropriately secured same-origin integration. The API is experimental, and only one voice is active at a time.

## Start the engine and page

Install the runtime and private voice package first; see [Setup](SETUP.md) and [Voice packages](PHASE6.md). Activate the project's Python environment and run:

```powershell
python -m voxcpm_realtime.voice_manager --mode offline
```

Open http://127.0.0.1:8767/ on this computer. Choose an installed voice and click **Load voice**. Loading or switching starts a backend and may take 15 seconds or more. Enter text, click **Generate speech**, then use the audio controls or download link. Audio is generated locally, watermarked and verified before the page receives it. It is a complete WAV response rather than progressive playback. The page also works with `--mode streaming`, but this endpoint still waits for the full response. In streaming backend mode, bad-case retries are disabled, as required by audio.cpp.

For a terminal-only workflow, open a second window in the project directory:

```powershell
python -m voxcpm_realtime.voices list
python -m voxcpm_realtime.voices select step150
python -m voxcpm_realtime.voices status
```

## HTTP API

The default manager address is `http://127.0.0.1:8767`. The manager must be running and a voice must be selected before synthesis. The manager serializes voice switching and managed GPU requests; it keeps one voice loaded at a time. A different voice can be selected via the page or `POST /api/voices/select` with `{"voice":"<id>"}`. The manager validates a package first and attempts to restore the prior voice after a runtime load failure; restoration is not guaranteed. `/health` reports the active voice and mode, and `/api/voices` lists package validation results.

`POST /api/synthesize` accepts JSON:

```json
{"voice":"step150","text":"Hello from a local application.","seed":35}
```

`text` is required, must contain non-whitespace text, and is limited to 2,000 characters. `voice` is optional; when present it must match the active voice. `seed` is optional, defaults to 35, and must be an integer from 0 to 4294967295. On success the response body is a 48 kHz, mono PCM16 WAV with `Content-Type: audio/wav`, `X-Audio-Watermark: perth-implicit`, and `Cache-Control: no-store`. The server applies Perth on CPU and runs the local detector on the encoded WAV before returning it. Missing dependencies, watermarking errors, and detector failure return an error rather than raw audio. Detection is not a guarantee of robustness against edits or every transformation.

Example from PowerShell in a second window:

```powershell
python -m voxcpm_realtime.voices select step150
$body = @{ voice = 'step150'; text = 'Hello from a local application.'; seed = 35 } | ConvertTo-Json
Invoke-WebRequest -Uri 'http://127.0.0.1:8767/api/synthesize' `
  -Method Post -ContentType 'application/json' -Body $body -OutFile 'outputs/api-watermarked.wav'
Start-Process 'outputs/api-watermarked.wav'
```

Python example using the standard library:

```python
import json
from urllib.request import Request, urlopen

payload = json.dumps({'voice': 'step150', 'text': 'Hello from a local application.'}).encode('utf-8')
request = Request('http://127.0.0.1:8767/api/synthesize', data=payload,
                  headers={'Content-Type': 'application/json'}, method='POST')
with urlopen(request, timeout=300) as response:
    audio = response.read()
with open('speech.wav', 'wb') as output:
    output.write(audio)
```

Errors are JSON and do not contain audio: `422` for invalid input, `409` for an active-voice mismatch, `503` for no selected voice or a watermark failure, `502` for a backend error or incomplete stream, and `504` for a backend timeout. `/docs` exposes the FastAPI schema. The backend response is limited to 64 MiB before watermarking. A client should handle timeouts and failures instead of assuming that a response will always contain audio.

## Streaming and raw endpoints

The HTTP endpoint above returns a full WAV, even if the backend is configured for streaming. For experimental progressive playback use the Python `stream_watermarked_speech` client documented in [Watermarked streaming](PHASE6_STREAMING.md). It buffers and checks individual segments; a late failure may interrupt playback after some audio has already been heard, and a short excerpt may not detect reliably.

The manager also relays `/v1/audio/speech`, and audio.cpp exposes its own backend port, but those return **unwatermarked** audio. They are not substitutes for `/api/synthesize` when using `step150`. Direct backend requests also bypass the manager's switch lock.

## Local smoke test

On 2026-11-21 the real RTX 3050 setup returned HTTP 200 and a 3.2-second 48 kHz WAV from `/api/synthesize` in both offline and streaming backend modes. Perth independently scored each saved WAV 1.0. This was a small local test, not a quality or robustness guarantee. The page's HTML was served successfully, but an automated visual browser check was not completed in that environment.
