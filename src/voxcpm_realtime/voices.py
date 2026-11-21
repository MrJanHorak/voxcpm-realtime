"""Private local voice packages. Existing voice.json metadata is never overwritten."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from .inference import EXPECTED_REVISION
from .validator import inspect_voice

MANIFEST = 'voxcpm-voice.json'
ID_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$')


def load_package(folder):
    root = Path(folder).expanduser().resolve()
    try:
        data = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError(f'Cannot read package manifest in {root}: {exc}') from exc
    if not isinstance(data, dict) or data.get('schema_version') != 1:
        raise ValueError('Unsupported voice package schema')
    voice_id = data.get('id')
    if not isinstance(voice_id, str) or not ID_PATTERN.fullmatch(voice_id) or voice_id != root.name:
        raise ValueError('Package id must match the directory name and use letters, digits, _ or -')
    if data.get('base_model_revision') != EXPECTED_REVISION:
        raise ValueError('Package base model revision does not match the supported revision')
    if data.get('watermark') != 'perth-implicit':
        raise ValueError('Package must explicitly require perth-implicit watermarking')
    ref = data.get('reference')
    if ref not in (None, 'reference.wav'):
        raise ValueError('Reference must be reference.wav or null')
    result = inspect_voice(root)
    if not result['compatible']:
        raise ValueError('Voice assets failed validation: ' + '; '.join(f['message'] for f in result['findings'] if f['level'] == 'error'))
    if ref and not (root / ref).is_file():
        raise ValueError('Package reference.wav is missing')
    return {'id': voice_id, 'directory': str(root), 'reference': str(root / ref) if ref else None,
            'base_model_revision': data['base_model_revision'], 'watermark': data['watermark'],
            'runtime_verified': False}


def initialize(folder):
    root = Path(folder).expanduser().resolve()
    existing = json.loads((root / 'voice.json').read_text(encoding='utf-8'))
    if existing.get('base_model', {}).get('revision') != EXPECTED_REVISION:
        raise ValueError('Existing voice metadata does not identify the supported base revision')
    watermark = existing.get('output', {}).get('watermark', '')
    if not isinstance(watermark, str) or not all(word in watermark.lower() for word in ('perth', 'implicit')):
        raise ValueError('Existing voice metadata does not explicitly require Perth watermarking')
    manifest = {'schema_version': 1, 'id': root.name, 'base_model_revision': EXPECTED_REVISION,
                'reference': 'reference.wav' if (root / 'reference.wav').is_file() else None,
                'watermark': 'perth-implicit'}
    if not ID_PATTERN.fullmatch(root.name):
        raise ValueError('Directory name is not a valid voice id')
    result = inspect_voice(root)
    if not result['compatible']:
        raise ValueError('Voice assets failed validation')
    with (root / MANIFEST).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    return load_package(root)


def list_packages(root):
    root = Path(root).expanduser()
    if not root.is_dir():
        return []
    results = []
    for folder in sorted(root.iterdir()):
        if folder.is_dir() and (folder / MANIFEST).is_file():
            try:
                results.append(load_package(folder))
            except ValueError as exc:
                results.append({'directory': str(folder.resolve()), 'error': str(exc)})
    return results


def main():
    p = argparse.ArgumentParser(description='Manage private local voice packages')
    sub = p.add_subparsers(dest='command', required=True)
    init = sub.add_parser('init'); init.add_argument('folder')
    check = sub.add_parser('check'); check.add_argument('folder')
    listing = sub.add_parser('list'); listing.add_argument('--root', default='voices')
    args = p.parse_args()
    try:
        result = initialize(args.folder) if args.command == 'init' else load_package(args.folder) if args.command == 'check' else list_packages(args.root)
        print(json.dumps(result, indent=2))
    except (ValueError, OSError) as exc:
        p.exit(1, f'Error: {exc}\n')

if __name__ == '__main__':
    main()
