> **Phase 6 correction:** The existing step150 metadata requires a Perth implicit watermark. The raw server output described here does not apply it. The Python streaming client now has an [experimental watermarked path](PHASE6_STREAMING.md); the [buffered client](PHASE6.md) remains available.

# Phase 5 — Streaming speech

This experimental integration uses the audio.cpp server configured in **streaming** mode, with the same local Q8 model, LoRA and reference as Phase 4. Do not start a second GPU server at the same time on the 4 GiB GPU. Stop any existing offline server, then run:

```powershell
.\scripts\start_audio_cpp.ps1 -Mode streaming
```

From a second PowerShell window with the project's `.venv` active:

```powershell
python -m voxcpm_realtime.stream --text 'Hello, this is a live streaming test.' --output outputs/stream-test.wav
```

The client now watermarks and verifies buffered segments before playing PCM on the default device and also writes a WAV; see [watermarked streaming](PHASE6_STREAMING.md) for current behavior and timing. `--output` is optional and refuses to overwrite an existing file. Use `--no-play --output outputs/stream-test.wav` to measure receiving without playing. If live playback dependencies are missing, install them in the virtual environment with `pip install -e ".[watermark,playback]"`; on this machine sounddevice 0.5.6 was installed. The output file, if specified, may contain incomplete audio if a request fails; a successful command requires the server completion event and `[DONE]` marker. Press Ctrl+C to interrupt the client; server-side cancellation and GPU resource release on disconnect have not been verified. Stop the server separately with Ctrl+C.

The streaming server serves the same `/v1/audio/speech` endpoint as the offline server but requires `stream_format: sse`, `response_format: pcm` and `options.retry_badcase: false`. Offline and streaming are different sessions: the launcher defaults to offline, and `-Mode streaming` selects streaming. Retrying bad generations is unavailable in this streaming path. This is a localhost service with no authentication; it is not suitable for untrusted networks. The server response is unwatermarked; the updated high-level client adds the watermark as described above.

For low-level Python integration, `voxcpm_realtime.stream.stream_speech(text)` yields raw unwatermarked audio as signed little-endian 16-bit mono PCM chunks at 48 kHz as they arrive. The iterator owns an HTTP connection; exhaust it for success or call `close()` on cancellation. The client detects malformed audio, explicit SSE errors and truncated streams; HTTP 200 alone is not proof of success. These audio properties are specific to the configured VoxCPM2 backend, not a general contract for every audio.cpp model. Playback writes chunks directly to sounddevice and can stutter if generation or the audio device falls behind; there is no jitter buffer or playback cancellation protocol.

## Measurements on 2026-11-21

RTX 3050 Laptop 4 GiB, Q8 base, the trained `step150` LoRA and reference, audio.cpp 0.9.5 CUDA, seed 35. The text was `Hello, this is a test of the voice engine. Streaming lets us hear the voice while it is being generated.` The no-play client received 11 chunks totaling 6.24 seconds. In a fresh server, first audio arrived 12.27 seconds after client start and the complete stream arrived at 20.24 seconds, including model/adapter loading, reference encoding and initial GPU work. A second request with the same text and warm session received first audio in 0.576 seconds and completed in 3.186 seconds (receive-time RTF 0.51). The warm request had reference-cache hits. These are single local measurements, not guarantees for other text or concurrent workloads.

A separate live-playback invocation for a different short sentence successfully opened the default device, received six chunks totaling 2.88 seconds, and exited successfully. Its reported first-audio time was 1.10 seconds and its elapsed time 3.81 seconds, including device initialization and blocking playback writes. This confirms the playback path executed but does not independently verify what was audible or whether there were glitches. The user previously found the Phase 4 offline samples sounded good; the Phase 5 streaming output has not received a separate listening assessment. The WAV outputs were kept locally under ignored `outputs/`.
