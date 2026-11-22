"""Manage one local audio.cpp process and serialize voice switches with requests."""
from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
import json
import io
import wave
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

import httpx
from fastapi import FastAPI, HTTPException, Request as FastAPIRequest
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from .client import VoiceClient
from .stream import SAMPLE_RATE, StreamError, read_events
from .runtime_paths import backend_paths
from .voices import ID_PATTERN, list_packages, load_package


class SwitchError(RuntimeError):
    def __init__(self, message, restored=False):
        super().__init__(message)
        self.restored = restored


class BackendManager:
    """Own exactly one backend child; never attach to or terminate other processes."""
    def __init__(self, root, mode='streaming', backend_port=18766, startup_timeout=180, voices_root=None):
        self.root = Path(root).resolve()
        self.voices_root = Path(voices_root).resolve() if voices_root else self.root / 'voices'
        self.mode = mode
        self.port = backend_port
        self.timeout = startup_timeout
        self.process = None
        self.voice = None
        self._config_path = None
        self._log_path = self.root / 'outputs' / 'voice-manager-backend.log'

    @property
    def url(self):
        return f'http://127.0.0.1:{self.port}'

    def status(self):
        process = self.process
        voice = self.voice
        ready = process is not None and voice is not None and process.poll() is None and process is self.process
        return {'voice': voice if ready else None, 'ready': ready,
                'mode': self.mode, 'backend_port': self.port}

    def validate(self, voice_id):
        if not isinstance(voice_id, str) or not ID_PATTERN.fullmatch(voice_id):
            raise ValueError('Invalid voice id')
        return load_package(self.voices_root / voice_id)

    def _check_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', self.port))
            except OSError as exc:
                raise RuntimeError(f'Backend port {self.port} is already in use; stop the other server first') from exc

    def _start(self, package):
        exe, runtime, model = backend_paths(self.root)
        if not exe.is_file() or not runtime.is_dir() or not model.is_file():
            raise RuntimeError('audio.cpp executable, CUDA runtime, or local GGUF is missing; run python -m voxcpm_realtime.install_runtime')
        self._check_port()
        entry = {'id': package['id'], 'family': 'voxcpm2', 'path': str(model), 'task': 'tts',
                 'mode': self.mode, 'session_options': {'voxcpm2.lora': package['directory']}}
        if package['reference']:
            entry['default_voice_preset'] = {'voice_ref': package['reference']}
        config = {'host': '127.0.0.1', 'port': self.port, 'backend': 'cuda', 'device': 0,
                  'threads': 4, 'lazy_load': True, 'max_loaded_models': 1, 'models': [entry]}
        output = self.root / 'outputs'
        output.mkdir(parents=True, exist_ok=True)
        fd, filename = tempfile.mkstemp(prefix='voice-manager-', suffix='.json', dir=output)
        self._config_path = Path(filename)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(config, f)
        env = os.environ.copy()
        env['PATH'] = str(runtime) + os.pathsep + env.get('PATH', '')
        with self._log_path.open('a', encoding='utf-8') as log:
            self.process = subprocess.Popen([str(exe), '--config', filename, '--log'], cwd=self.root,
                                            env=env, stdout=log, stderr=subprocess.STDOUT,
                                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f'Backend exited during startup (code {self.process.returncode}); see {self._log_path}')
            try:
                with urlopen(self.url + '/health', timeout=1) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError, OSError):
                time.sleep(0.2)
        else:
            raise RuntimeError('Backend did not become available before timeout')
        # A listening HTTP server is not a loaded model. Force a real request and
        # consume its completion, including SSE errors, before declaring ready.
        request = {'model': package['id'], 'input': 'Ready.', 'seed': 35}
        if self.mode == 'streaming':
            request.update({'response_format': 'pcm', 'stream_format': 'sse',
                            'options': {'retry_badcase': False}})
        req = Request(self.url + '/v1/audio/speech', data=json.dumps(request).encode(),
                      headers={'Content-Type': 'application/json'}, method='POST')
        with urlopen(req, timeout=self.timeout) as response:
            if self.mode == 'streaming':
                if response.headers.get_content_type() != 'text/event-stream':
                    raise RuntimeError('Backend warmup returned the wrong content type')
                for _ in read_events(response):
                    pass
            elif response.headers.get_content_type() not in ('audio/wav', 'audio/x-wav') or not response.read(44).startswith(b'RIFF'):
                raise RuntimeError('Backend warmup did not return WAV audio')
        self.voice = package['id']

    def stop(self):
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                # This is the exact child we started, not a process found by name.
                process.kill()
                process.wait(timeout=10)
        self.process = None
        self.voice = None
        if self._config_path is not None:
            self._config_path.unlink(missing_ok=True)
            self._config_path = None

    def select(self, voice_id):
        package = self.validate(voice_id)  # Preflight before disturbing current voice.
        previous = self.voice if self.process is not None and self.process.poll() is None else None
        if previous == voice_id:
            return self.status()
        old_package = self.validate(previous) if previous else None
        self.stop()
        try:
            self._start(package)
        except Exception as exc:
            self.stop()
            if old_package:
                try:
                    self._start(old_package)
                except Exception as restore_exc:
                    self.stop()
                    raise SwitchError(f'Voice load failed: {exc}. Previous voice could not be restored: {restore_exc}') from exc
                raise SwitchError(f'Voice load failed; previous voice restored: {exc}', restored=True) from exc
            raise SwitchError(f'Voice load failed: {exc}') from exc
        return self.status()


class SelectRequest(BaseModel):
    voice: str


class WatermarkedSynthesisRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    voice: str | None = None
    seed: int = Field(default=35, ge=0, le=4294967295)


def create_app(manager, watermarker=None):
    lock = asyncio.Lock()
    watermarker = watermarker if watermarker is not None else VoiceClient()

    @asynccontextmanager
    async def lifespan(app):
        yield
        async with lock:
            await asyncio.to_thread(manager.stop)

    app = FastAPI(title='VoxCPM Local Voice Manager', lifespan=lifespan)

    @app.get('/', response_class=HTMLResponse, include_in_schema=False)
    async def index():
        page = Path(__file__).parent / 'static' / 'index.html'
        return HTMLResponse(page.read_text(encoding='utf-8'), headers={'Cache-Control': 'no-store'})

    @app.post('/api/synthesize', response_class=Response, responses={200: {'content': {'audio/wav': {}}}})
    async def synthesize(body: WatermarkedSynthesisRequest):
        if not body.text.strip():
            raise HTTPException(422, detail='Text must not be empty')
        async with lock:
            status = manager.status()
            if not status['ready']:
                raise HTTPException(503, detail='Select a voice first')
            if body.voice is not None and body.voice != status['voice']:
                raise HTTPException(409, detail='Selected voice does not match the active voice')
            payload = {'model': status['voice'], 'input': body.text, 'seed': body.seed}
            if status['mode'] == 'streaming':
                payload.update({'response_format': 'pcm', 'stream_format': 'sse',
                                'options': {'retry_badcase': False}})
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(manager.timeout), trust_env=False) as client:
                    async with client.stream('POST', manager.url + '/v1/audio/speech', json=payload) as upstream:
                        if upstream.status_code != 200:
                            raise HTTPException(502, detail=f'Voice backend returned HTTP {upstream.status_code}')
                        content_type = upstream.headers.get('content-type', '').split(';', 1)[0].strip().lower()
                        expected = ('text/event-stream',) if status['mode'] == 'streaming' else ('audio/wav', 'audio/x-wav')
                        if content_type not in expected:
                            raise HTTPException(502, detail='Voice backend returned an unexpected audio format')
                        data = bytearray()
                        async for part in upstream.aiter_bytes():
                            data.extend(part)
                            if len(data) > 64 * 1024 * 1024:
                                raise HTTPException(502, detail='Voice backend response exceeds the size limit')
            except httpx.TimeoutException as exc:
                raise HTTPException(504, detail='Voice backend timed out') from exc
            except httpx.HTTPError as exc:
                raise HTTPException(502, detail='Voice backend could not be reached') from exc
        if status['mode'] == 'streaming':
            try:
                chunks = read_events(bytes(data).splitlines(keepends=True))
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wav:
                    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(SAMPLE_RATE)
                    for chunk in chunks:
                        wav.writeframesraw(chunk)
                raw_wav = buffer.getvalue()
            except (StreamError, UnicodeError, ValueError) as exc:
                raise HTTPException(502, detail=f'Voice backend returned an incomplete stream: {exc}') from exc
        else:
            raw_wav = bytes(data)
        try:
            marked = await asyncio.to_thread(watermarker.watermark_wav, raw_wav)
        except Exception as exc:
            raise HTTPException(503, detail='Watermarking or verification failed; no audio returned') from exc
        return Response(content=marked, media_type='audio/wav',
                        headers={'Cache-Control': 'no-store', 'X-Audio-Watermark': 'perth-implicit'})

    @app.get('/health')
    async def health():
        return manager.status()

    @app.get('/api/voices')
    async def voices():
        return await asyncio.to_thread(list_packages, manager.voices_root)

    @app.post('/api/voices/select')
    async def select(body: SelectRequest):
        async with lock:
            try:
                task = asyncio.create_task(asyncio.to_thread(manager.select, body.voice))
                try:
                    return await asyncio.shield(task)
                except asyncio.CancelledError:
                    # Keep the lock until the worker finishes changing processes.
                    try:
                        await task
                    finally:
                        raise
            except ValueError as exc:
                raise HTTPException(422, detail=str(exc)) from exc
            except SwitchError as exc:
                raise HTTPException(503, detail={'message': str(exc), 'previous_restored': exc.restored}) from exc

    @app.post('/v1/audio/speech')
    async def speech(request: FastAPIRequest):
        body = await request.body()
        try:
            data = json.loads(body)
            if not isinstance(data, dict):
                raise ValueError()
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(400, detail='Expected a JSON object')
        await lock.acquire()
        client = httpx.AsyncClient(timeout=httpx.Timeout(manager.timeout), trust_env=False)
        response = None
        try:
            status = manager.status()
            if not status['ready']:
                raise HTTPException(503, detail='Select a voice first')
            if data.get('model') != status['voice']:
                raise HTTPException(409, detail=f"Active voice is {status['voice']}; select the requested voice first")
            upstream = client.build_request('POST', manager.url + '/v1/audio/speech', content=body,
                                            headers={'Content-Type': 'application/json', 'Accept': request.headers.get('accept', '*/*')})
            response = await client.send(upstream, stream=True)
        except Exception:
            if response is not None:
                await response.aclose()
            await client.aclose()
            lock.release()
            raise

        async def relay():
            try:
                async for chunk in response.aiter_raw():
                    yield chunk
            finally:
                await response.aclose()
                await client.aclose()
                lock.release()
        return StreamingResponse(relay(), status_code=response.status_code,
                                 media_type=response.headers.get('content-type'))

    return app


def main():
    parser = argparse.ArgumentParser(description='Run the local single-voice manager')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--backend-port', type=int, default=18766)
    parser.add_argument('--mode', choices=('offline', 'streaming'), default='streaming')
    parser.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--voices-root', help='Optional directory containing private voice packages')
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(BackendManager(args.root, mode=args.mode, backend_port=args.backend_port, voices_root=args.voices_root)),
                host='127.0.0.1', port=args.port)

if __name__ == '__main__':
    main()
