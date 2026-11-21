import json
import wave
import numpy as np
from safetensors.numpy import save_file
from voxcpm_realtime.validator import inspect_voice

def test_missing_folder(tmp_path):
    result = inspect_voice(tmp_path / "missing")
    assert not result["compatible"]

def test_valid_adapter(tmp_path):
    (tmp_path / "lora_config.json").write_text(json.dumps({"r": 32, "lora_alpha": 32}))
    save_file({"layer.lora_A.weight": np.zeros((2, 2), dtype=np.float32)}, str(tmp_path / "lora_weights.safetensors"))
    result = inspect_voice(tmp_path)
    assert result["compatible"]
    assert result["tensor_count"] == 1
    assert not result["runtime_verified"]

def test_bad_weights(tmp_path):
    (tmp_path / "lora_config.json").write_text("{}")
    (tmp_path / "lora_weights.safetensors").write_bytes(b"invalid")
    assert not inspect_voice(tmp_path)["compatible"]

def test_reference(tmp_path):
    (tmp_path / "lora_config.json").write_text("{}")
    save_file({"lora_A": np.zeros((1,), dtype=np.float32)}, str(tmp_path / "lora_weights.safetensors"))
    with wave.open(str(tmp_path / "reference.wav"), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(b"\x00\x00" * 16000)
    assert inspect_voice(tmp_path)["reference"]["duration_seconds"] == 1.0
