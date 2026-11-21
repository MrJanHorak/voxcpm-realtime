import sys
import threading
import time
import types
import numpy as np
import pytest
from fastapi.testclient import TestClient
from voxcpm_realtime.manager import ModelManager
from voxcpm_realtime import api


class FakeTensor:
    def squeeze(self, _): return self
    def cpu(self): return self
    def numpy(self): return np.zeros(480, dtype=np.float32)

class FakeTTS:
    sample_rate = 48000
    def __init__(self):
        self.builds = 0
        self.active = 0
        self.peak = 0
    def build_prompt_cache(self, **kwargs):
        self.builds += 1
        return {"mode": "reference"}
    def generate_with_prompt_cache(self, **kwargs):
        self.active += 1
        self.peak = max(self.peak, self.active)
        time.sleep(0.02)
        self.active -= 1
        return FakeTensor(), None, None


def test_serializes_and_invalidates_reference_cache(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(manual_seed=lambda _: None))
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"a")
    manager = ModelManager(tmp_path, tmp_path, reference)
    tts = FakeTTS()
    manager._model = types.SimpleNamespace(tts_model=tts)
    threads = [threading.Thread(target=manager.generate, args=("hello",)) for _ in range(3)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert tts.peak == 1
    assert tts.builds == 1
    reference.write_bytes(b"changed")
    manager.generate("hello")
    assert tts.builds == 2


def test_api_returns_wav_and_handles_blank(monkeypatch):
    class FakeManager:
        loaded = True
        warmed = False
        def generate(self, text, seed):
            if not text.strip(): raise ValueError("Text must not be empty")
            return np.zeros(480, dtype=np.float32), 48000
    monkeypatch.setattr(api, "manager", FakeManager())
    client = TestClient(api.app)
    response = client.post("/api/synthesize", json={"text": "hello"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content[:4] == b"RIFF"
    assert client.post("/api/synthesize", json={"text": " "}).status_code == 422


def test_model_loaded_once_and_partial_adapter_rejected(tmp_path, monkeypatch):
    (tmp_path / 'lora_config.json').write_text('{"r": 2}')
    (tmp_path / 'lora_weights.safetensors').write_bytes(b'fake')
    calls = []
    class Config:
        model_fields = {'r': None}
        def __init__(self, **kwargs): pass
    class Model:
        tts_model = FakeTTS()
        def load_lora(self, path):
            calls.append(path)
            return ['loaded'], []
    class Factory:
        @staticmethod
        def from_pretrained(*args, **kwargs): return Model()
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(manual_seed=lambda _: None))
    monkeypatch.setitem(sys.modules, 'voxcpm', types.SimpleNamespace(VoxCPM=Factory))
    monkeypatch.setitem(sys.modules, 'voxcpm.model', types.ModuleType('voxcpm.model'))
    monkeypatch.setitem(sys.modules, 'voxcpm.model.voxcpm', types.SimpleNamespace(LoRAConfig=Config))
    manager = ModelManager(tmp_path, tmp_path)
    manager.generate('hello')
    manager.generate('hello')
    assert len(calls) == 1
    assert manager.loaded
    monkeypatch.setattr(Model, 'load_lora', lambda self, path: (['loaded'], ['skipped']))
    failed = ModelManager(tmp_path, tmp_path)
    with pytest.raises(RuntimeError, match='incomplete'):
        failed.generate('hello')
    assert not failed.loaded
