from pathlib import Path
import httpx
import pytest
from fastapi.testclient import TestClient
from voxcpm_realtime.voice_manager import BackendManager, SwitchError, create_app


class FakeProcess:
    def poll(self): return None


class FakeManager(BackendManager):
    def __init__(self, root):
        super().__init__(root)
        self.started = []
        self.stopped = 0
        self.fail = set()
    def validate(self, voice_id):
        if voice_id == 'invalid': raise ValueError('Invalid package')
        return {'id': voice_id, 'directory': voice_id, 'reference': None}
    def _start(self, package):
        self.started.append(package['id'])
        self.process = FakeProcess()
        if package['id'] in self.fail: raise RuntimeError('bad adapter')
        self.voice = package['id']
    def stop(self):
        self.stopped += 1
        self.process = None
        self.voice = None


def test_invalid_selection_does_not_stop_existing_voice(tmp_path):
    manager = FakeManager(tmp_path)
    manager.select('first')
    before = manager.stopped
    with pytest.raises(ValueError): manager.select('invalid')
    assert manager.voice == 'first' and manager.stopped == before


def test_runtime_failure_restores_prior_voice(tmp_path):
    manager = FakeManager(tmp_path)
    manager.select('first')
    manager.fail.add('second')
    with pytest.raises(SwitchError) as exc:
        manager.select('second')
    assert exc.value.restored
    assert manager.status()['voice'] == 'first'
    assert manager.started == ['first', 'second', 'first']


def test_failed_restore_reports_no_active_voice(tmp_path):
    manager = FakeManager(tmp_path)
    manager.select('first')
    manager.fail.update({'first', 'second'})
    with pytest.raises(SwitchError) as exc:
        manager.select('second')
    assert not exc.value.restored and manager.status()['voice'] is None


def test_manager_relay_and_voice_mismatch(tmp_path, monkeypatch):
    manager = FakeManager(tmp_path)
    app = create_app(manager)
    original_client = httpx.AsyncClient
    class BytesStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'RIFFsample'
    def handler(request):
        assert request.url.path == '/v1/audio/speech'
        return httpx.Response(200, stream=BytesStream(), headers={'content-type':'audio/wav'})
    monkeypatch.setattr(httpx, 'AsyncClient', lambda **kwargs: original_client(transport=httpx.MockTransport(handler), **kwargs))
    with TestClient(app) as client:
        assert client.post('/v1/audio/speech', json={'model': 'first'}).status_code == 503
        assert client.post('/api/voices/select', json={'voice': 'first'}).json()['voice'] == 'first'
        assert client.post('/v1/audio/speech', json={'model': 'second'}).status_code == 409
        response = client.post('/v1/audio/speech', json={'model': 'first', 'input': 'hi'})
        assert response.status_code == 200 and response.content == b'RIFFsample'
        assert response.headers['content-type'] == 'audio/wav'
    assert manager.status()['ready'] is False


def test_switch_waits_until_stream_finishes(tmp_path, monkeypatch):
    import asyncio
    manager = FakeManager(tmp_path)
    manager.select('first')
    app = create_app(manager)
    original_client = httpx.AsyncClient
    async def exercise():
        emitted = asyncio.Event()
        finish = asyncio.Event()
        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                emitted.set()
                yield b'first'
                await finish.wait()
                yield b'last'
        def handler(request):
            return httpx.Response(200, stream=SlowStream(), headers={'content-type': 'text/event-stream'})
        monkeypatch.setattr(httpx, 'AsyncClient', lambda **kw: original_client(transport=httpx.MockTransport(handler), **kw))
        async with original_client(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            speech = asyncio.create_task(client.post('/v1/audio/speech', json={'model':'first'}))
            await asyncio.wait_for(emitted.wait(), 2)
            switch = asyncio.create_task(client.post('/api/voices/select', json={'voice':'second'}))
            await asyncio.sleep(0.05)
            assert manager.voice == 'first' and not switch.done()
            finish.set()
            assert (await asyncio.wait_for(speech, 2)).content == b'firstlast'
            assert (await asyncio.wait_for(switch, 2)).json()['voice'] == 'second'
    asyncio.run(exercise())


def test_manager_refuses_port_in_use(tmp_path):
    import socket
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); sock.listen()
        manager = BackendManager(tmp_path, backend_port=sock.getsockname()[1])
        with pytest.raises(RuntimeError, match='already in use'):
            manager._check_port()


def test_start_failure_removes_only_owned_config(tmp_path, monkeypatch):
    import voxcpm_realtime.voice_manager as module
    manager = BackendManager(tmp_path)
    output = tmp_path / 'outputs'
    (output / 'audio-cpp-bin').mkdir(parents=True)
    (output / 'audio-cpp-bin' / 'audiocpp_server.exe').write_bytes(b'fake')
    (output / 'audio-cpp-runtime').mkdir()
    (output / 'voxcpm2-q8-local.gguf').write_bytes(b'fake')
    unrelated = output / 'unrelated.json'
    unrelated.write_text('keep')
    monkeypatch.setattr(manager, '_check_port', lambda: None)
    def fail_spawn(*args, **kwargs): raise OSError('spawn failed')
    monkeypatch.setattr(module.subprocess, 'Popen', fail_spawn)
    monkeypatch.setattr(manager, 'validate', lambda voice: {'id':voice, 'directory':'voice', 'reference':None})
    with pytest.raises(SwitchError, match='spawn failed'):
        manager.select('first')
    assert unrelated.read_text() == 'keep'
    assert list(output.glob('voice-manager-*.json')) == []
    assert manager.process is None


def test_backend_is_not_ready_until_warmup_completes(tmp_path):
    manager = FakeManager(tmp_path)
    manager.process = FakeProcess()
    assert manager.status()['ready'] is False
    assert manager.status()['voice'] is None


def test_real_child_process_restarts_and_rolls_back(tmp_path, monkeypatch):
    import socket
    import subprocess
    import sys
    import voxcpm_realtime.voice_manager as module
    # A CPU-only backend simulates HTTP/SSE and a runtime-only adapter failure.
    stub = tmp_path / 'backend_stub.py'
    stub.write_text('''import base64, json, sys\nfrom http.server import BaseHTTPRequestHandler, HTTPServer\nconfig=json.load(open(sys.argv[1], encoding="utf-8")); voice=config["models"][0]["id"]\nclass Handler(BaseHTTPRequestHandler):\n def do_GET(self):\n  self.send_response(200); self.end_headers(); self.wfile.write(b"ok")\n def do_POST(self):\n  self.rfile.read(int(self.headers.get("Content-Length", "0")))\n  if voice == "bad":\n   self.send_response(500); self.end_headers(); self.wfile.write(b"invalid adapter"); return\n  events=b\'data: {"type":"speech.audio.delta","audio":"AAA="}\\n\\ndata: {"type":"speech.audio.done"}\\n\\ndata: [DONE]\\n\\n\'\n  self.send_response(200); self.send_header("Content-Type","text/event-stream"); self.send_header("Content-Length",str(len(events))); self.end_headers(); self.wfile.write(events)\nHTTPServer(("127.0.0.1", config["port"]),Handler).serve_forever()\n''', encoding='utf-8')
    out = tmp_path / 'outputs'
    (out / 'audio-cpp-bin').mkdir(parents=True)
    (out / 'audio-cpp-bin' / 'audiocpp_server.exe').write_bytes(b'fake')
    (out / 'audio-cpp-runtime').mkdir()
    (out / 'voxcpm2-q8-local.gguf').write_bytes(b'fake')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    manager = BackendManager(tmp_path, backend_port=port, startup_timeout=10)
    # The stub server reuses its own port; the independent port-guard test covers occupied listeners.
    monkeypatch.setattr(manager, '_check_port', lambda: None)
    monkeypatch.setattr(manager, 'validate', lambda voice: {'id':voice, 'directory':'voice', 'reference':None})
    original_popen = subprocess.Popen
    monkeypatch.setattr(module.subprocess, 'Popen', lambda command, **kw: original_popen([sys.executable, str(stub), command[2]], **kw))
    try:
        assert manager.select('first')['voice'] == 'first'
        first_pid = manager.process.pid
        assert manager.select('second')['voice'] == 'second'
        assert manager.process.pid != first_pid
        with pytest.raises(SwitchError) as exc:
            manager.select('bad')
        assert exc.value.restored and manager.status()['voice'] == 'second'
        with TestClient(create_app(manager)) as client:
            response = client.post('/v1/audio/speech', json={'model':'second','input':'test'})
            assert response.status_code == 200
            assert b'speech.audio.delta' in response.content and b'[DONE]' in response.content
    finally:
        manager.stop()
    assert not list(out.glob('voice-manager-*.json'))


def test_dedicated_backend_does_not_require_outputs_in_advance(tmp_path, monkeypatch):
    import voxcpm_realtime.voice_manager as module
    root = tmp_path
    bin_dir = root / '.runtime/audio-cpp-v0.9.5/bin'; bin_dir.mkdir(parents=True)
    (bin_dir / 'audiocpp_server.exe').write_bytes(b'fake')
    (root / '.runtime/audio-cpp-v0.9.5/runtime').mkdir()
    model = root / 'models/VoxCPM2-GGUF/voxcpm2-q8_0.gguf'; model.parent.mkdir(parents=True)
    model.write_bytes(b'fake')
    manager = BackendManager(root)
    monkeypatch.setattr(manager, '_check_port', lambda: None)
    monkeypatch.setattr(manager, 'validate', lambda voice: {'id':voice, 'directory':'voice', 'reference':None})
    seen = []
    def fail_spawn(command, **kwargs):
        seen.append(command[0]); raise OSError('simulated')
    monkeypatch.setattr(module.subprocess, 'Popen', fail_spawn)
    with pytest.raises(SwitchError, match='simulated'):
        manager.select('first')
    assert seen == [str(bin_dir / 'audiocpp_server.exe')]
    assert (root / 'outputs').is_dir()
