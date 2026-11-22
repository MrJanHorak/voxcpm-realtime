import hashlib
import json
import zipfile
from pathlib import Path
import pytest
from voxcpm_realtime import install_runtime as install
from voxcpm_realtime.runtime_paths import backend_paths


def test_extract_rejects_traversal_and_duplicate(tmp_path):
    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('../escape.txt', 'bad')
    with pytest.raises(RuntimeError, match='Unsafe'):
        install.safe_extract(archive, tmp_path / 'dest')
    assert not (tmp_path / 'escape.txt').exists()
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('A.txt', 'a'); z.writestr('a.txt', 'b')
    with pytest.raises(RuntimeError, match='Duplicate'):
        install.safe_extract(archive, tmp_path / 'dest')


def test_install_component_checks_archive_and_existing_files(tmp_path, monkeypatch):
    archives = tmp_path / 'downloads'; archives.mkdir()
    archive = archives / 'package.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        z.writestr('tool.exe', b'correct')
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    monkeypatch.setitem(install.ARCHIVES, 'test', ('official.zip','package.zip',digest))
    monkeypatch.setitem(install.CRITICAL, 'test', ('tool.exe',))
    target = install.install_component(tmp_path, 'test', archives)
    assert (target / 'tool.exe').read_bytes() == b'correct'
    assert install.install_component(tmp_path, 'test', archives) == target
    (target / 'tool.exe').write_bytes(b'modified')
    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        install.install_component(tmp_path, 'test', archives)


def test_base_checksum_mismatch_fails(tmp_path, monkeypatch):
    (tmp_path / 'file').write_bytes(b'wrong')
    monkeypatch.setattr(install, 'BASE_HASHES', {'file': hashlib.sha256(b'right').hexdigest()})
    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        install.verify_base(tmp_path)


def test_model_install_is_transactional_and_verified(tmp_path, monkeypatch):
    base = tmp_path / 'base'; base.mkdir()
    for name in install.SIDECARS: (base / name).write_text('{}')
    (base / 'model.safetensors').write_bytes(b'base')
    (base / 'audiovae.pth').write_bytes(b'vae')
    bin_dir = tmp_path / 'bin'; bin_dir.mkdir()
    runtime = tmp_path / 'runtime'; runtime.mkdir()
    monkeypatch.setattr(install, 'verify_base', lambda _: None)
    monkeypatch.setattr(install, 'convert_audio_vae', lambda source, dest: dest.write_bytes(b'vae'))
    def run(command, **kwargs):
        if '--output' in command: Path(command[command.index('--output') + 1]).write_bytes(b'gguf')
    monkeypatch.setattr(install.subprocess, 'run', run)
    output = install.install_model(tmp_path, base, bin_dir, runtime)
    assert output.read_bytes() == b'gguf'
    assert install.install_model(tmp_path, base, bin_dir, runtime) == output
    output.write_bytes(b'corrupt')
    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        install.install_model(tmp_path, base, bin_dir, runtime)


def test_backend_paths_prefer_complete_dedicated_install(tmp_path):
    assert backend_paths(tmp_path)[0] == tmp_path / 'outputs/audio-cpp-bin/audiocpp_server.exe'
    bin_dir = tmp_path / '.runtime/audio-cpp-v0.9.5/bin'; bin_dir.mkdir(parents=True)
    (bin_dir / 'audiocpp_server.exe').write_bytes(b'')
    (tmp_path / '.runtime/audio-cpp-v0.9.5/runtime').mkdir()
    model = tmp_path / 'models/VoxCPM2-GGUF/voxcpm2-q8_0.gguf'; model.parent.mkdir(parents=True)
    model.write_bytes(b'')
    assert backend_paths(tmp_path)[2] == model


def test_rejects_wrong_archive_without_install(tmp_path, monkeypatch):
    archives = tmp_path / 'archives'; archives.mkdir()
    archive = archives / 'package.zip'
    with zipfile.ZipFile(archive, 'w') as z: z.writestr('tool.exe', b'value')
    monkeypatch.setitem(install.ARCHIVES, 'test', ('official.zip', 'package.zip', '0' * 64))
    monkeypatch.setitem(install.CRITICAL, 'test', ('tool.exe',))
    with pytest.raises(RuntimeError, match='SHA-256 mismatch'):
        install.install_component(tmp_path, 'test', archives)
    assert not (tmp_path / '.runtime/audio-cpp-v0.9.5/test').exists()


def test_failed_conversion_does_not_publish_model(tmp_path, monkeypatch):
    import subprocess
    base = tmp_path / 'base'; base.mkdir()
    for name in install.SIDECARS: (base / name).write_text('{}')
    monkeypatch.setattr(install, 'verify_base', lambda _: None)
    monkeypatch.setattr(install, 'convert_audio_vae', lambda source, dest: dest.write_bytes(b'vae'))
    def fail(command, **kwargs): raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(install.subprocess, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        install.install_model(tmp_path, base, tmp_path / 'bin', tmp_path / 'runtime')
    assert not (tmp_path / 'models/VoxCPM2-GGUF').exists()
    assert not list((tmp_path / 'models').glob('.voxcpm-convert-*'))


def test_download_fallback_uses_verified_bytes(tmp_path, monkeypatch):
    import io
    content = io.BytesIO()
    with zipfile.ZipFile(content, 'w') as z: z.writestr('tool.exe', b'downloaded')
    payload = content.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setitem(install.ARCHIVES, 'test', ('official.zip', 'local.zip', digest))
    monkeypatch.setitem(install.CRITICAL, 'test', ('tool.exe',))
    urls = []
    def fake_urlopen(url, timeout):
        urls.append(url)
        return io.BytesIO(payload)
    monkeypatch.setattr(install, 'urlopen', fake_urlopen)
    target = install.install_component(tmp_path, 'test', tmp_path / 'missing')
    assert urls == [install.RELEASE + 'official.zip']
    assert (target / 'tool.exe').read_bytes() == b'downloaded'
