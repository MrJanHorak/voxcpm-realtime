# VoxCPM Real-Time Voice Engine

A local Windows voice engine for trained VoxCPM2 LoRA voices. It uses a pinned audio.cpp Q8 backend and can return Perth-watermarked WAV audio through a local API. The browser test page lets you load a voice, enter text and play or download the result. Voice recordings, adapters, model weights and generated audio stay on your computer and are excluded from Git.

## Quick start

The following assumes the dedicated runtime, matching base model and a private `step150` voice package are already installed. For a new installation, see [Setup](docs/SETUP.md) and [Voice packages](docs/PHASE6.md).

1. Open PowerShell in the project folder and activate `.venv`.
2. Start the local manager:

   ```powershell
   python -m voxcpm_realtime.voice_manager --mode offline
   ```

3. Open **http://127.0.0.1:8767/** on the same computer. Select `step150`, click **Load voice**, enter text, then click **Generate speech**. Use the audio controls to listen or download the WAV.

The page and `/api/synthesize` buffer the complete output before returning it. The manager also supports `--mode streaming`; the watermarked API remains buffered in either mode. To test the separate experimental live-streaming Python client, see [Watermarked streaming](docs/PHASE6_STREAMING.md).

## For application developers

Use `POST http://127.0.0.1:8767/api/synthesize` for a complete 48 kHz mono WAV with the Perth watermark applied and checked. Start the manager and select a voice first. See [API integration](docs/API.md) for complete HTTP and Python examples, errors and streaming details.

**Important:** `/v1/audio/speech`, the underlying audio.cpp server and older native endpoints return raw audio without applying the watermark. The `step150` package explicitly requires the Perth implicit watermark. Use `/api/synthesize` or the provided watermarking clients for this voice.

## Current scope

- One active voice at a time; switching restarts the backend. Switching between distinct trained voices has not yet been tested.
- The local manager and test page have no authentication and are intended for localhost, not direct exposure to a network or the internet.
- Watermarked live streaming is experimental. Segment detection, short-excerpt detection and failure behavior have limitations described in [Watermarked streaming](docs/PHASE6_STREAMING.md).
- The tested hardware is an NVIDIA RTX 3050 Laptop with 4 GB VRAM. Loading a voice takes time, and other GPU workloads can affect availability and speed.
- The voice package validator performs static checks; it cannot prove a voice is compatible with a model or assess generated speech quality.
- The local detector check does not guarantee a watermark survives all edits or transformations. Label generated audio appropriately and use voice assets only with permission.

## Documentation

- [Dedicated Windows runtime setup](docs/SETUP.md)
- [Application API and browser test page](docs/API.md)
- [Private voice packages and buffered watermarking](docs/PHASE6.md)
- [Voice manager and switching](docs/PHASE6_MANAGER.md)
- [Experimental watermarked streaming](docs/PHASE6_STREAMING.md)
- [Earlier native and Q8 benchmarks](docs/PHASE3.md)

The separate `voiceover-automation` training repository is not modified by this project.
