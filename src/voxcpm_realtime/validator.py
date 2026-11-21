"""Static validation only: never deserializes pickle or executes model code."""
from __future__ import annotations
import json
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from safetensors import safe_open

@dataclass
class Finding:
    level: str
    code: str
    message: str

def inspect_voice(folder: str | Path) -> dict:
    root = Path(folder).expanduser().resolve()
    findings: list[Finding] = []
    def add(level: str, code: str, message: str):
        findings.append(Finding(level, code, message))
    if not root.is_dir():
        add("error", "missing_directory", f"Voice directory not found: {root}")
        return {"voice": str(root), "compatible": False, "findings": [asdict(x) for x in findings]}
    config_path = root / "lora_config.json"
    weights_path = root / "lora_weights.safetensors"
    config = None
    if not config_path.is_file():
        add("error", "missing_config", "Expected lora_config.json")
    else:
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict):
                raise ValueError("LoRA config must be a JSON object")
        except (OSError, ValueError) as exc:
            add("error", "invalid_config", str(exc))
    tensor_count = 0
    tensor_examples = []
    if not weights_path.is_file():
        add("error", "missing_weights", "Expected lora_weights.safetensors")
    else:
        try:
            with safe_open(str(weights_path), framework="np", device="cpu") as f:
                keys = list(f.keys())
                tensor_count = len(keys)
                tensor_examples = keys[:8]
                if not keys:
                    add("error", "empty_weights", "No tensors found")
                elif not any("lora" in k.lower() for k in keys):
                    add("warning", "unrecognized_keys", "No tensor names contain 'lora'; verify adapter format")
        except Exception as exc:
            add("error", "invalid_weights", f"Unable to inspect safetensors header: {exc}")
    reference = root / "reference.wav"
    audio = None
    if reference.exists():
        try:
            with wave.open(str(reference), "rb") as wav:
                audio = {"sample_rate": wav.getframerate(), "channels": wav.getnchannels(), "sample_width_bytes": wav.getsampwidth(), "duration_seconds": round(wav.getnframes()/wav.getframerate(), 3)}
                if audio["channels"] != 1:
                    add("warning", "reference_channels", "Mono reference audio is recommended")
        except (wave.Error, OSError, ZeroDivisionError) as exc:
            add("error", "invalid_reference", str(exc))
        if not (root / "transcript.txt").is_file():
            add("warning", "missing_transcript", "Reference audio provided without transcript.txt")
    elif (root / "transcript.txt").exists():
        add("warning", "orphan_transcript", "Transcript provided without reference.wav")
    manifest_path = root / "voice.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("voice.json must contain an object")
        except (OSError, ValueError) as exc:
            add("error", "invalid_manifest", str(exc))
    if config is not None and tensor_count:
        add("info", "static_validation", "Adapter files are readable; base-model revision and runtime loading remain unverified")
    return {"voice": str(root), "compatible": not any(f.level == "error" for f in findings), "runtime_verified": False, "config": config, "manifest": manifest, "tensor_count": tensor_count, "tensor_examples": tensor_examples, "reference": audio, "findings": [asdict(x) for x in findings]}
