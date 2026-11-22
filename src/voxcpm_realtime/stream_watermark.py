"""Perth watermarking of individual 48 kHz mono PCM16 stream chunks."""
from __future__ import annotations
import numpy as np


class ChunkWatermarkError(RuntimeError):
    """A chunk could not be watermarked and verified; it must not be emitted."""


class PerthChunkWatermarker:
    """CPU watermarker. Each returned PCM chunk has passed local Perth detection."""
    def __init__(self, watermarker=None, threshold=0.5):
        if watermarker is None:
            import perth
            if perth.PerthImplicitWatermarker is None:
                raise ChunkWatermarkError('Perth implicit watermarker is unavailable')
            watermarker = perth.PerthImplicitWatermarker(device='cpu')
        self.watermarker = watermarker
        self.threshold = threshold

    def process(self, pcm):
        if not pcm or len(pcm) % 2:
            raise ChunkWatermarkError('Expected nonempty PCM16 samples')
        signal = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
        try:
            marked = np.asarray(self.watermarker.apply_watermark(signal, sample_rate=48000), dtype=np.float32)
            if marked.shape != signal.shape or not np.isfinite(marked).all():
                raise ChunkWatermarkError('Watermarker returned invalid audio')
            encoded = np.rint(np.clip(marked, -1.0, 1.0 - 1.0 / 32768.0) * 32768.0).astype('<i2').tobytes()
            decoded = np.frombuffer(encoded, dtype='<i2').astype(np.float32) / 32768.0
            score = np.asarray(self.watermarker.get_watermark(decoded, 48000, round=False))
            if not score.size or not np.isfinite(score).all() or float(score.mean()) < self.threshold:
                raise ChunkWatermarkError('Perth detection failed for a streaming chunk')
            return encoded
        except ChunkWatermarkError:
            raise
        except Exception as exc:
            raise ChunkWatermarkError(f'Cannot watermark or verify streaming chunk: {exc}') from exc


def watermark_segments(chunks, marker, *, segment_samples=48000, reserve_samples=48000, max_samples=480000):
    """Buffer at least one second behind each output to protect a short final tail.

    A failing segment is extended and retried; failures never emit raw audio.
    A late failure cannot retract audio already yielded to a listener.
    """
    if not (0 < segment_samples <= max_samples and reserve_samples > 0):
        raise ValueError('Invalid segment limits')
    pending = bytearray()
    wanted = segment_samples * 2
    step = segment_samples * 2
    reserve = reserve_samples * 2
    limit = max_samples * 2
    for chunk in chunks:
        if not chunk or len(chunk) % 2:
            raise ChunkWatermarkError('Invalid source PCM16 chunk')
        pending.extend(chunk)
        while len(pending) >= wanted + reserve:
            try:
                result = marker.process(bytes(pending[:wanted]))
            except ChunkWatermarkError:
                wanted += step
                if wanted > limit:
                    raise ChunkWatermarkError('Could not verify a streaming segment within the buffer limit')
                continue
            del pending[:wanted]
            wanted = step
            yield result
    if pending:
        if len(pending) > limit + reserve:
            raise ChunkWatermarkError('Final streaming segment exceeds the buffer limit')
        yield marker.process(bytes(pending))
