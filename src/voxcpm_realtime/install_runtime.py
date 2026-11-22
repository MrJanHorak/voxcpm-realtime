"""Pinned Windows CUDA runtime installation and exact-base Q8 model conversion."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
from urllib.request import urlopen
import zipfile

from .inference import EXPECTED_REVISION

VERSION = 'v0.9.5'
RELEASE = 'https://github.com/0xShug0/audio.cpp/releases/download/v0.9.5/'
ARCHIVES = {
    'bin': ('audio-v0.9.5-bin-windows-x64-cuda12.4.zip', 'audio-v0.9.5-cuda12.4.zip', '81f38a0a62935bdf4b2605541a21d50223b2e12b661887031705dd8a369f844b'),
    'runtime': ('audio-v0.9.5-cudart-windows-x64-cuda12.4.zip', 'audio-v0.9.5-cudart12.4.zip', 'fa55dfaff1d4b5c6e8fac460d64f380b9df76cd46435f187c39b05700f623a7e'),
}
BASE_HASHES = {
    'model.safetensors': 'f7f964cfa9da23653baec6e6f7750719977ad944ed9f95fe52fe3a620506891d',
    'audiovae.pth': '94b5d51e107e0507d4acc976cfdadb64edd6fd06d1f751dadbf2fd1594274bf1',
    'config.json': '405f0dcd92f7feba6011ed4eac5c8d4f74cba9712f07fd5cfa3063bbdd95402c',
    'tokenizer_config.json': 'e78a3ebb48a0b9437efd1823b6b726c823da89e49dd8bcc90c02419d9baa772b',
    'tokenizer.json': 'f8984687e4a92a3503d521396d454b7d68e9fdaab2a0288eb3536c7c1aa4bc20',
    'special_tokens_map.json': '068594063e37662c02b21acf42ebb334ef6a74fb810e68a2368f88f08351de76',
}
SIDECARS = ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'special_tokens_map.json')
CRITICAL = {'bin': ('audiocpp_server.exe', 'audiocpp_cli.exe', 'audiocpp_gguf.exe', 'ggml-cuda.dll', 'model_specs/voxcpm2.json'),
            'runtime': ('cudart64_12.dll', 'cublas64_12.dll', 'cublasLt64_12.dll', 'cufft64_11.dll')}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_hash(path, expected):
    if not path.is_file():
        raise RuntimeError(f'Missing required file: {path}')
    if sha256(path) != expected:
        raise RuntimeError(f'SHA-256 mismatch: {path}')


def safe_extract(archive, destination):
    """Extract only relative files, rejecting links, traversal, and duplicates."""
    seen = set()
    with zipfile.ZipFile(archive) as z:
        for item in z.infolist():
            name = item.filename
            path = PurePosixPath(name)
            if not name or '\\' in name or path.is_absolute() or '..' in path.parts or ':' in name:
                raise RuntimeError(f'Unsafe archive entry: {name}')
            if stat.S_ISLNK(item.external_attr >> 16):
                raise RuntimeError(f'Archive link is not allowed: {name}')
            key = str(path).rstrip('/').casefold()
            if key in seen:
                raise RuntimeError(f'Duplicate archive entry: {name}')
            seen.add(key)
        z.extractall(destination)


def check_runtime_component(target, kind):
    try:
        manifest = json.loads((target / 'install.json').read_text(encoding='utf-8'))
        if manifest['archive_sha256'] != ARCHIVES[kind][2]:
            raise ValueError('archive hash')
        for rel in CRITICAL[kind]:
            verify_hash(target / rel, manifest['critical_sha256'][rel])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f'Existing {kind} installation is incomplete or invalid: {target}') from exc


def install_component(root, kind, archive_dir):
    target = root / '.runtime' / ('audio-cpp-' + VERSION) / kind
    if target.exists():
        check_runtime_component(target, kind)
        print(f'Using verified existing {kind}: {target}', flush=True)
        return target
    official, local, expected = ARCHIVES[kind]
    parent = target.parent
    if not target.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Installation target escapes project directory')
    parent.mkdir(parents=True, exist_ok=True)
    # TemporaryDirectory chooses its own child under the already-resolved workspace root.
    with tempfile.TemporaryDirectory(prefix='.install-', dir=parent) as work_name:
        work = Path(work_name).resolve()
        if work.parent != parent.resolve():
            raise RuntimeError('Unsafe temporary directory')
        archive = archive_dir / local
        if not archive.is_file():
            archive = archive_dir / official
        if not archive.is_file():
            archive = work / official
            print(f'Downloading official {kind} archive...', flush=True)
            with urlopen(RELEASE + official, timeout=60) as response, archive.open('xb') as f:
                shutil.copyfileobj(response, f, length=1024 * 1024)
        print(f'Checking {kind} archive SHA-256...', flush=True)
        verify_hash(archive, expected)
        stage = work / 'files'; stage.mkdir()
        safe_extract(archive, stage)
        hashes = {rel: sha256(stage / rel) for rel in CRITICAL[kind]}
        (stage / 'install.json').write_text(json.dumps({'version': VERSION, 'archive_sha256': expected,
                                                        'critical_sha256': hashes}, indent=2) + '\n', encoding='utf-8')
        # No replace: preserve any concurrently-created or user-managed installation.
        stage.rename(target)
    check_runtime_component(target, kind)
    print(f'Installed {kind}: {target}', flush=True)
    return target


def verify_base(base):
    for name, expected in BASE_HASHES.items():
        print(f'Checking base {name}...', flush=True)
        verify_hash(base / name, expected)


def convert_audio_vae(source, dest):
    import torch
    from safetensors.torch import load_file, save_file
    payload = torch.load(source, map_location='cpu', weights_only=True)
    state = payload.get('state_dict', payload) if isinstance(payload, dict) else None
    if not isinstance(state, dict) or not state or any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in state.items()):
        raise RuntimeError('Unexpected AudioVAE checkpoint contents')
    tensors = {k: v.detach().cpu().contiguous() for k, v in state.items()}
    save_file(tensors, str(dest))
    reloaded = load_file(str(dest), device='cpu')
    if set(reloaded) != set(tensors) or any(not torch.equal(v, reloaded[k]) for k, v in tensors.items()):
        raise RuntimeError('Converted AudioVAE tensors differ from the source')


def check_model(target):
    try:
        info = json.loads((target / 'provenance.json').read_text(encoding='utf-8'))
        if info['base_revision'] != EXPECTED_REVISION or info['base_sha256'] != BASE_HASHES or info['release'] != VERSION:
            raise ValueError('provenance mismatch')
        verify_hash(target / 'voxcpm2-q8_0.gguf', info['gguf_sha256'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f'Existing converted model is incomplete or invalid: {target}') from exc


def install_model(root, base, bin_dir, runtime_dir):
    target = root / 'models' / 'VoxCPM2-GGUF'
    if target.exists():
        check_model(target)
        print(f'Using verified existing model: {target}', flush=True)
        return target / 'voxcpm2-q8_0.gguf'
    verify_base(base)
    parent = target.parent
    if not target.resolve().is_relative_to(root.resolve()):
        raise RuntimeError('Installation target escapes project directory')
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.voxcpm-convert-', dir=parent) as work_name:
        work = Path(work_name).resolve()
        if work.parent != parent.resolve():
            raise RuntimeError('Unsafe temporary directory')
        stage = work / 'model'; stage.mkdir()
        sidecars = work / 'sidecars'; sidecars.mkdir()
        for name in SIDECARS:
            shutil.copyfile(base / name, sidecars / name)
        vae = work / 'audiovae.safetensors'
        print('Converting and checking AudioVAE tensors...', flush=True)
        convert_audio_vae(base / 'audiovae.pth', vae)
        gguf = stage / 'voxcpm2-q8_0.gguf'
        command = [str(bin_dir / 'audiocpp_gguf.exe'), '--input', f'weights={base / "model.safetensors"}',
                   '--input', f'audiovae_weights={vae}', '--root', str(sidecars), '--output', str(gguf),
                   '--type', 'q8_0', '--family', 'voxcpm2', '--model-spec', str(bin_dir / 'model_specs' / 'voxcpm2.json'),
                   '--keep-type', 'audiovae_weights/*=f16', '--exclude-prefix', 'audiovae_weights/decoder.sr_bin_boundaries']
        env = os.environ.copy(); env['PATH'] = str(runtime_dir) + os.pathsep + env.get('PATH', '')
        print('Converting local base model to Q8 GGUF...', flush=True)
        subprocess.run(command, check=True, env=env, cwd=root)
        if not gguf.is_file() or gguf.stat().st_size == 0:
            raise RuntimeError('Converter did not produce a GGUF')
        subprocess.run([str(bin_dir / 'audiocpp_gguf.exe'), '--inspect', str(gguf)], check=True,
                       env=env, cwd=root, stdout=subprocess.DEVNULL)
        info = {'base_revision': EXPECTED_REVISION, 'base_sha256': BASE_HASHES, 'release': VERSION,
                'type': 'q8_0', 'audiovae_type': 'f16', 'gguf_sha256': sha256(gguf)}
        (stage / 'provenance.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
        stage.rename(target)
    check_model(target)
    print(f'Installed model: {target}', flush=True)
    return target / 'voxcpm2-q8_0.gguf'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', default='models/VoxCPM2')
    parser.add_argument('--archives', default='outputs', help='Existing release archives; otherwise download official files')
    args = parser.parse_args()
    if os.name != 'nt':
        parser.error('The pinned runtime package is for Windows x64')
    root = Path(__file__).resolve().parents[2]
    base = Path(args.base_model).expanduser().resolve()
    archive_dir = Path(args.archives).expanduser().resolve()
    try:
        bin_dir = install_component(root, 'bin', archive_dir)
        runtime_dir = install_component(root, 'runtime', archive_dir)
        install_model(root, base, bin_dir, runtime_dir)
    except (OSError, RuntimeError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        parser.exit(1, f'Installation failed: {exc}\n')
    print('Setup complete. Start the voice manager or launcher from the project directory.')

if __name__ == '__main__':
    main()
