import numpy as np
import pytest
from voxcpm_realtime.stream_watermark import PerthChunkWatermarker, ChunkWatermarkError, watermark_segments
from voxcpm_realtime import stream


class FakePerth:
    def __init__(self, score=0.9): self.score = score
    def apply_watermark(self, signal, sample_rate): return signal + 0.01
    def get_watermark(self, signal, sample_rate, round=False): return self.score


def test_pcm_is_watermarked_and_checked_after_encoding():
    pcm = np.zeros(100, dtype='<i2').tobytes()
    output = PerthChunkWatermarker(FakePerth()).process(pcm)
    assert len(output) == len(pcm)
    assert np.frombuffer(output, dtype='<i2').mean() > 0
    with pytest.raises(ChunkWatermarkError, match='detection failed'):
        PerthChunkWatermarker(FakePerth(0.1)).process(pcm)
    with pytest.raises(ChunkWatermarkError, match='PCM16'):
        PerthChunkWatermarker(FakePerth()).process(b'\x00')


def test_segments_retry_without_leaking_and_keep_tail():
    class RequiresLonger:
        def process(self, chunk):
            if len(chunk) < 8: raise ChunkWatermarkError('too short')
            return b'Z' * len(chunk)
    # Segment is 2 samples; an initial rejection extends it to 4.
    result = list(watermark_segments([b'a' * 4, b'b' * 4, b'c' * 4, b'd' * 12], RequiresLonger(),
                                     segment_samples=2, reserve_samples=2, max_samples=10))
    assert result == [b'Z' * 8, b'Z' * 8, b'Z' * 8]


def test_failed_final_segment_is_not_emitted():
    class FailFinal:
        def process(self, chunk):
            if b'x' in chunk: raise ChunkWatermarkError('cannot verify')
            return b'Z' * len(chunk)
    it = watermark_segments([b'a' * 4, b'b' * 4, b'x' * 4], FailFinal(),
                            segment_samples=2, reserve_samples=2, max_samples=8)
    assert next(it) == b'Z' * 4
    with pytest.raises(ChunkWatermarkError): list(it)


def test_watermarked_stream_closes_source_on_cancel(monkeypatch):
    closed = []
    def source(*args, **kwargs):
        try:
            yield b'a' * 8
            yield b'b' * 8
        finally:
            closed.append(True)
    class Marker:
        def process(self, chunk): return chunk
    monkeypatch.setattr(stream, 'stream_speech', source)
    # Replace sizes to exercise cancellation without generating seconds of audio.
    import voxcpm_realtime.stream_watermark as wm
    original = wm.watermark_segments
    monkeypatch.setattr(wm, 'watermark_segments', lambda chunks, marker: original(chunks, marker, segment_samples=2, reserve_samples=2, max_samples=20))
    it = stream.stream_watermarked_speech('test', watermarker=Marker())
    assert next(it) == b'a' * 4
    it.close()
    assert closed == [True]
