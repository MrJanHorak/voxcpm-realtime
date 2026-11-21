"""Native VoxCPM2 one-shot inference. Heavy dependencies are imported only when used."""
from __future__ import annotations

import json
from pathlib import Path

EXPECTED_REVISION = "7936080b41163874f38610f46f438e2c0757f9fd"


def read_lora_config(checkpoint: str | Path, config_type):
    """Use saved rank/alpha instead of the library defaults."""
    path = Path(checkpoint) / "lora_config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("LoRA config must be a JSON object")
    data = data.get("lora_config", data)
    if not isinstance(data, dict):
        raise ValueError("lora_config must be an object")
    fields = config_type.model_fields
    values = {key: value for key, value in data.items() if key in fields}
    if not values:
        raise ValueError("No recognized LoRA configuration fields; refusing to use defaults")
    return config_type(**values)


def synthesize(*, model_dir: str | Path, voice_dir: str | Path, text: str,
               output: str | Path, reference_wav: str | Path | None = None,
               seed: int | None = 35) -> dict:
    """Load a matching local base model + adapter, render one WAV, then release model."""
    if not text.strip():
        raise ValueError("Text must not be empty")
    base = Path(model_dir).expanduser().resolve()
    voice = Path(voice_dir).expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Base model directory not found: {base}")
    if not (voice / "lora_weights.safetensors").is_file():
        raise FileNotFoundError(f"Missing adapter weights: {voice}")
    if not (voice / "lora_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter config: {voice}")
    if reference_wav is not None and not Path(reference_wav).is_file():
        raise FileNotFoundError(f"Reference WAV not found: {reference_wav}")

    try:
        import torch
        import soundfile as sf
        from voxcpm import VoxCPM
        from voxcpm.model.voxcpm import LoRAConfig
    except ImportError as exc:
        raise RuntimeError('Install the inference extras: pip install -e ".[inference]" and a compatible CUDA PyTorch build') from exc

    config = read_lora_config(voice, LoRAConfig)
    if seed is not None:
        torch.manual_seed(seed)
    model = None
    try:
        model = VoxCPM.from_pretrained(
            str(base), load_denoiser=False, optimize=False,
            local_files_only=True, lora_config=config, lora_weights_path=str(voice)
        )
        options = {"reference_wav_path": str(reference_wav)} if reference_wav else {}
        audio = model.generate(text=text, **options)
        dest = Path(output).expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        sample_rate = int(model.tts_model.sample_rate)
        sf.write(str(dest), audio, sample_rate)
        return {"output": str(dest), "sample_rate": sample_rate, "voice": str(voice),
                "reference_used": bool(reference_wav), "engine": "voxcpm2", "streaming": False}
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
