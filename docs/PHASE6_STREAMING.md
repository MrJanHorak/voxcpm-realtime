# Phase 6 — Experimental watermarked streaming

The Phase 5 Python streaming client now applies the required Perth implicit watermark on CPU and checks each emitted segment after PCM16 encoding. The audio.cpp server and the lower-level `stream_speech()` function still return **raw unwatermarked** audio; they are not a watermark enforcement boundary. Use the high-level client below or `stream_watermarked_speech()` for the experimental watermarked path. The buffered [Phase 6 client](PHASE6.md) remains the fallback.

Stop other GPU servers and start the configured package in streaming mode:

```powershell
.\scripts\start_audio_cpp.ps1 -VoiceId step150 -Mode streaming
```

In another window with `.venv` active:

```powershell
python -m voxcpm_realtime.stream --model step150 `
  --text 'Hello, this is a watermarked streaming test.' `
  --output outputs/watermarked-stream.wav
```

The command plays audio as it is received; `--output` additionally saves a new WAV and will not overwrite an existing file. `--no-play` requires an output path and measures receiving without playing. Install `pip install -e ".[watermark,playback]"` if needed; the established inference environment must also provide PyTorch and Perth's processing dependencies. The watermarker loads once per process on CPU. A new CLI process loads it again on each invocation.

`stream_watermarked_speech()` yields 48 kHz, signed PCM16 mono bytes for applications. It accepts an optional reusable `PerthChunkWatermarker` instance; close the iterator on cancellation. The implementation accumulates at least one second per segment and reserves another second to join a short tail. If a segment fails detection it is extended and retried up to a bounded buffer. Every emitted segment is watermarked and locally scored at least 0.5 after PCM16 conversion. If the final segment fails, the client raises an error rather than returning unverified bytes. Audio emitted earlier may already have been heard or saved and cannot be recalled; a WAV left by a failed command is incomplete. The method does not guarantee detection of every short excerpt or robustness to edits, transformations, compression or resampling. Watermarking may alter sound quality or produce clicks at segment joins. A local detector score is not a compliance guarantee.

A user can use a persistent watermarker to avoid repeated initialization:

```python
from voxcpm_realtime.stream import stream_watermarked_speech
from voxcpm_realtime.stream_watermark import PerthChunkWatermarker
marker = PerthChunkWatermarker()
for pcm in stream_watermarked_speech('Hello there.', watermarker=marker):
    # Send PCM to a 48 kHz mono PCM16 player.
    pass
```

## Test results, 2026-11-21

On the RTX 3050 Laptop with the existing Q8 model and `step150` reference, an initial 11-chunk experiment showed that testing float output alone was misleading: the first 0.16-second chunk scored 0.976 before PCM16 conversion and 0.026 afterward. Several quiet segments and a short final tail also failed detection. The resulting implementation buffers and checks encoded segments instead of accepting the raw chunks.

Three warm test requests using a preloaded CPU watermarker and GPU session produced 6.24, 2.40 and 10.88 seconds of audio. Their first verified audio arrived after 1.475, 1.078 and 1.081 seconds; complete reception took 3.022, 1.106 and 4.538 seconds. A separate CLI process took 4.30 seconds to first audio for the 6.24-second test, including Python/Perth initialization, and 5.88 seconds to complete. These are individual local measurements; network, text, silence and device conditions affect results.

Independent detection on the three complete PCM16 WAVs scored 0.932, 0.957 and 0.984, respectively. Five one-second sections independently scored between 0.747 and 1.0; the last 0.24 seconds considered alone scored 0.0016, even though the 1.24-second final emitted segment passed. Sample-to-sample jumps at one-second joins ranged from zero to 0.0056 in normalized amplitude; this does not prove that joins are inaudible. A separate live-playback command completed successfully through the default device, but audible quality, latency and glitches have not been independently assessed. No stress tests, second voice, cancellation-under-load tests, or watermark robustness studies have been completed. Tests were saved only under Git-ignored `outputs/`.
