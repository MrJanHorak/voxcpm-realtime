# Phase 6 — Local voice manager

The manager provides one local service for selecting voice packages and routing speech requests to **one owned audio.cpp backend process at a time**. It supports the previously tested watermarked Python clients; the manager's relay and backend return raw, unwatermarked audio, so continue to use the watermarked clients for `step150`. The streaming watermark limitations in [Phase 6 streaming](PHASE6_STREAMING.md) still apply.

## Start and select

Stop any separately started audio.cpp or native GPU server first. In one PowerShell window with `.venv` active, from the repository root:

```powershell
python -m voxcpm_realtime.voice_manager --mode streaming
```

In a second window:

```powershell
python -m voxcpm_realtime.voices list
python -m voxcpm_realtime.voices select step150
python -m voxcpm_realtime.voices status
python -m voxcpm_realtime.stream --url http://127.0.0.1:8767 --model step150 `
  --text 'Hello, this is a managed voice test.' --output outputs/managed-stream.wav
```

A selection waits for a real backend warmup, so the first selection and each switch can take time. The voice ID in a speech request must match the active voice. To use the buffered watermarked client instead, start the manager with `--mode offline` and use `python -m voxcpm_realtime.client --url http://127.0.0.1:8767 --model step150 ...`. A running manager has one mode; stop and restart it to change modes.

For another installed package, select its ID with `python -m voxcpm_realtime.voices select <id>` and pass the same ID as `--model`. Invalid packages fail static validation before the current voice is stopped. For a runtime load failure, the manager stops the failed backend and **attempts** to reload the previous voice. A successful rollback is reported; if rollback fails, no voice is reported ready and the error describes both failures. Restoration cannot be guaranteed, for example if GPU resources have become unavailable. The manager waits for managed speech responses to finish before switching.

The manager listens on `127.0.0.1:8767` and starts a backend on `127.0.0.1:18766` by default; override these with `--port` and `--backend-port`. It refuses an occupied backend port and does not attach to or terminate a process it did not start. Stop the manager with Ctrl+C; it normally stops its owned child at shutdown. A forced termination or crash can leave the backend running; check for that before restarting. Do not connect clients directly to the backend port: such requests bypass the manager's switch lock. Directly started servers and other GPU applications are not controlled by the manager, and it cannot ensure enough GPU memory if those applications are active. Both endpoints are local-only and unauthenticated.

The manager requires the existing Q8 model, audio.cpp executable and CUDA runtime under `outputs/` as described in Phase 4. Its backend log and temporary configuration are also under ignored `outputs/`; the temporary configuration is removed during normal shutdown. The local package and voice audio remain ignored by Git.

## Validation

Automated tests cover static preflight, rollback success and failure, routing, port conflicts and waiting for an in-flight managed stream. Static package validation is not proof that adapter weights match the model. A CPU-only simulated HTTP/SSE backend was also used to exercise actual child-process startup, switching, a runtime failure, rollback, and the relay.

On the RTX 3050 Laptop on 2026-11-21, a real streaming-mode manager selected `step150` in 13.55 seconds and switched to `step150_alias` in 15.03 seconds, including backend restart and warmup. A deliberately invalid adapter passed static preflight but failed at runtime; the manager returned HTTP 503, restored `step150_alias`, and reported it ready after 15.77 seconds. A watermarked streaming request through the restored manager returned 2.72 seconds of audio in two verified segments: first audio arrived in 1.673 seconds and reception completed in 1.808 seconds with a preloaded watermarker. Independent Perth detection on the saved WAV scored 0.967. These are individual local measurements, not guarantees. The manager and its backend exited on normal shutdown.

The two isolated test package IDs contain copies of the same voice; switching between distinct trained voices has not been tested. Streaming cancellation under load, recovery from forced manager termination, and manager crashes were not tested. The complete audio detector score does not establish detection for every short excerpt or after transformations.
