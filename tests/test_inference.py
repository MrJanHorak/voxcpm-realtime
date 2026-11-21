import json
from pathlib import Path
import pytest
from voxcpm_realtime.inference import read_lora_config, synthesize

class FakeConfig:
    model_fields = {"r": None, "lora_alpha": None}
    def __init__(self, **kwargs):
        self.values = kwargs

def test_reads_nested_config(tmp_path):
    (tmp_path / "lora_config.json").write_text(json.dumps({"lora_config": {"r": 32, "lora_alpha": 32, "ignored": True}}))
    assert read_lora_config(tmp_path, FakeConfig).values == {"r": 32, "lora_alpha": 32}

def test_rejects_unrecognized_config(tmp_path):
    (tmp_path / "lora_config.json").write_text('{"unknown": 1}')
    with pytest.raises(ValueError, match="No recognized"):
        read_lora_config(tmp_path, FakeConfig)

def test_missing_base_model(tmp_path):
    with pytest.raises(FileNotFoundError, match="Base model"):
        synthesize(model_dir=tmp_path / "missing", voice_dir=tmp_path,
                   text="hello", output=tmp_path / "out.wav")

def test_rejects_blank_text(tmp_path):
    with pytest.raises(ValueError, match="Text"):
        synthesize(model_dir=tmp_path, voice_dir=tmp_path,
                   text=" ", output=tmp_path / "out.wav")
