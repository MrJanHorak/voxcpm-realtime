"""Resolve the dedicated backend installation, with a legacy development fallback."""
from pathlib import Path


def backend_paths(root):
    root = Path(root)
    installed = root / '.runtime' / 'audio-cpp-v0.9.5'
    stable = (installed / 'bin' / 'audiocpp_server.exe', installed / 'runtime',
              root / 'models' / 'VoxCPM2-GGUF' / 'voxcpm2-q8_0.gguf')
    legacy = (root / 'outputs' / 'audio-cpp-bin' / 'audiocpp_server.exe',
              root / 'outputs' / 'audio-cpp-runtime', root / 'outputs' / 'voxcpm2-q8-local.gguf')
    if stable[0].is_file() and stable[1].is_dir() and stable[2].is_file():
        return stable
    return legacy
