import io
import json
import numpy as np
import pytest
import soundfile as sf
from safetensors.numpy import save_file
from voxcpm_realtime.inference import EXPECTED_REVISION
from voxcpm_realtime.voices import initialize, list_packages, load_package
from voxcpm_realtime.client import VoiceClient


def make_voice(tmp_path):
    root = tmp_path / 'test_voice'
    root.mkdir()
    (root / 'lora_config.json').write_text('{"lora_config":{"r":2}}')
    save_file({'layer.lora_A.weight': np.zeros((2, 2), dtype=np.float32)}, str(root / 'lora_weights.safetensors'))
    sf.write(root / 'reference.wav', np.zeros(16000, dtype=np.float32), 16000, subtype='PCM_16')
    (root / 'voice.json').write_text(json.dumps({'base_model': {'revision': EXPECTED_REVISION},
                                                  'output': {'watermark': 'Perth implicit watermark'}}))
    return root


def test_package_initialization_and_listing(tmp_path):
    folder = make_voice(tmp_path)
    before = (folder / 'voice.json').read_bytes()
    package = initialize(folder)
    assert package['id'] == 'test_voice'
    assert package['watermark'] == 'perth-implicit'
    assert list_packages(tmp_path)[0]['id'] == 'test_voice'
    assert (folder / 'voice.json').read_bytes() == before
    with pytest.raises(FileExistsError): initialize(folder)


def test_rejects_bad_revision_and_unrequired_watermark(tmp_path):
    folder = make_voice(tmp_path)
    manifest = folder / 'voxcpm-voice.json'
    manifest.write_text(json.dumps({'schema_version': 1, 'id': 'test_voice',
                                    'base_model_revision': 'wrong', 'watermark': 'perth-implicit'}))
    with pytest.raises(ValueError, match='revision'): load_package(folder)
    manifest.write_text(json.dumps({'schema_version': 1, 'id': 'test_voice',
                                    'base_model_revision': EXPECTED_REVISION, 'watermark': 'none'}))
    with pytest.raises(ValueError, match='watermark'): load_package(folder)


class Watermarker:
    def __init__(self, score): self.score = score
    def apply_watermark(self, audio, sample_rate): return audio + 0.01
    def get_watermark(self, audio, rate, round=False): return self.score


def test_watermark_client_fails_closed():
    source = io.BytesIO()
    sf.write(source, np.zeros(48000, dtype=np.float32), 48000, format='WAV')
    output = VoiceClient(watermarker=Watermarker(0.9)).watermark_wav(source.getvalue())
    marked, rate = sf.read(io.BytesIO(output))
    assert rate == 48000 and marked.mean() > 0.009
    with pytest.raises(RuntimeError, match='verification failed'):
        VoiceClient(watermarker=Watermarker(0.1)).watermark_wav(source.getvalue())
    with pytest.raises(ValueError, match='local'):
        VoiceClient(url='http://example.com')
