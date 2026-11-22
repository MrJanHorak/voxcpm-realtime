import base64
import io
import json
import wave
import httpx
from fastapi.testclient import TestClient
from voxcpm_realtime.voice_manager import BackendManager, create_app


class ReadyManager(BackendManager):
    def __init__(self, root, mode):
        super().__init__(root, mode=mode)
        self.active = 'step150'
    def status(self): return {'voice': self.active, 'ready': self.active is not None, 'mode': self.mode}
    def stop(self): pass


class FakeWatermarker:
    def __init__(self, failure=False): self.raw = None; self.failure = failure
    def watermark_wav(self, raw):
        self.raw = raw
        if self.failure: raise RuntimeError('Perth failed')
        return b'RIFFwatermarked'


def test_watermarked_offline_api(tmp_path, monkeypatch):
    manager = ReadyManager(tmp_path, 'offline'); marker = FakeWatermarker()
    app = create_app(manager, watermarker=marker)
    original = httpx.AsyncClient
    def handle(request):
        body = json.loads(request.content)
        assert body['model'] == 'step150' and body['input'] == 'hello'
        return httpx.Response(200, content=b'RIFFraw', headers={'content-type':'audio/wav'})
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original(transport=httpx.MockTransport(handle), **kw))
    with TestClient(app) as client:
        result = client.post('/api/synthesize', json={'voice':'step150','text':'hello'})
        assert result.status_code == 200 and result.content == b'RIFFwatermarked'
        assert result.headers['content-type'] == 'audio/wav'
        assert result.headers['x-audio-watermark'] == 'perth-implicit'
        assert marker.raw == b'RIFFraw'
        assert client.post('/api/synthesize', json={'voice':'other','text':'hello'}).status_code == 409
        assert client.post('/api/synthesize', json={'text':'   '}).status_code == 422
        assert client.post('/api/synthesize', json={'text':'a'*2001}).status_code == 422
        assert client.get('/').status_code == 200
        assert 'Voice Lab' in client.get('/').text


def test_streaming_backend_is_buffered_and_watermarked(tmp_path, monkeypatch):
    manager = ReadyManager(tmp_path, 'streaming'); marker = FakeWatermarker()
    app = create_app(manager, watermarker=marker); original = httpx.AsyncClient
    pcm = b'\x01\x00\x02\x00'
    def handle(request):
        body = json.loads(request.content)
        assert body['stream_format'] == 'sse' and body['options']['retry_badcase'] is False
        events = ('data: '+json.dumps({'type':'speech.audio.delta','audio':base64.b64encode(pcm).decode()})+'\n\n'
                  + 'data: {"type":"speech.audio.done"}\n\ndata: [DONE]\n\n')
        return httpx.Response(200, content=events.encode(), headers={'content-type':'text/event-stream'})
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original(transport=httpx.MockTransport(handle), **kw))
    with TestClient(app) as client:
        result = client.post('/api/synthesize', json={'text':'hello'})
        assert result.status_code == 200 and result.content == b'RIFFwatermarked'
        with wave.open(io.BytesIO(marker.raw), 'rb') as wav:
            assert wav.getframerate() == 48000 and wav.getnchannels() == 1 and wav.readframes(2) == pcm


def test_errors_never_return_raw_audio(tmp_path, monkeypatch):
    manager = ReadyManager(tmp_path, 'offline'); marker = FakeWatermarker(failure=True)
    app = create_app(manager, watermarker=marker); original = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, content=b'RIFFraw', headers={'content-type':'audio/wav'})), **kw))
    with TestClient(app) as client:
        result = client.post('/api/synthesize', json={'text':'hello'})
        assert result.status_code == 503 and b'RIFFraw' not in result.content
        manager.active = None
        assert client.post('/api/synthesize', json={'text':'hello'}).status_code == 503


def test_incomplete_stream_is_rejected(tmp_path, monkeypatch):
    manager = ReadyManager(tmp_path, 'streaming'); marker = FakeWatermarker()
    app = create_app(manager, watermarker=marker); original = httpx.AsyncClient
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original(transport=httpx.MockTransport(
        lambda req: httpx.Response(200, content=b'data: {"type":"speech.audio.delta","audio":"AAA="}\n\n', headers={'content-type':'text/event-stream'})), **kw))
    with TestClient(app) as client:
        assert client.post('/api/synthesize', json={'text':'hello'}).status_code == 502
        assert marker.raw is None
