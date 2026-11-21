"""Persistent, serialized VoxCPM2 inference for one configured voice."""
from __future__ import annotations

import re
import threading
import time
from pathlib import Path

from .inference import read_lora_config


class ModelManager:
    def __init__(self, model_dir, voice_dir, reference_wav=None):
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.voice_dir = Path(voice_dir).expanduser().resolve()
        self.reference_wav = Path(reference_wav).expanduser().resolve() if reference_wav else None
        self._lock = threading.Lock()
        self._model = None
        self._cache = None
        self._reference_signature = None
        self.load_seconds = None
        self.warmed = False

    @property
    def loaded(self):
        return self._model is not None

    def _load(self):
        if self._model is not None:
            return
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"Base model directory not found: {self.model_dir}")
        for name in ("lora_config.json", "lora_weights.safetensors"):
            if not (self.voice_dir / name).is_file():
                raise FileNotFoundError(f"Missing adapter file: {self.voice_dir / name}")
        from voxcpm import VoxCPM
        from voxcpm.model.voxcpm import LoRAConfig
        config = read_lora_config(self.voice_dir, LoRAConfig)
        start = time.perf_counter()
        model = VoxCPM.from_pretrained(str(self.model_dir), load_denoiser=False,
                                       optimize=False, local_files_only=True,
                                       lora_config=config)
        loaded, skipped = model.load_lora(str(self.voice_dir))
        if not loaded or skipped:
            raise RuntimeError(f"LoRA load incomplete: {len(loaded)} loaded, {len(skipped)} skipped")
        self.load_seconds = time.perf_counter() - start
        self._model = model

    def _conditioning(self):
        if self.reference_wav is None:
            return None
        stat = self.reference_wav.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if self._cache is None or signature != self._reference_signature:
            cache = self._model.tts_model.build_prompt_cache(reference_wav_path=str(self.reference_wav))
            self._cache = cache
            self._reference_signature = signature
        return self._cache

    def _generate(self, text, seed):
        import torch
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must not be empty")
        self._load()
        cache = self._conditioning()
        if seed is not None:
            torch.manual_seed(seed)
        # Match VoxCPM.generate defaults and its whitespace normalization.
        text = re.sub(r"\s+", " ", text.replace("\n", " "))
        wav, _, _ = self._model.tts_model.generate_with_prompt_cache(
            target_text=text, prompt_cache=cache, min_len=2, max_len=4096,
            inference_timesteps=10, cfg_value=2.0, retry_badcase=True,
            retry_badcase_max_times=3, retry_badcase_ratio_threshold=6.0)
        return wav.squeeze(0).cpu().numpy(), int(self._model.tts_model.sample_rate)

    def generate(self, text, seed=35):
        with self._lock:
            return self._generate(text, seed)

    def warmup(self, text="Hello, this is a test.", seed=35):
        with self._lock:
            self._generate(text, seed)
            self.warmed = True

    def write(self, text, output, seed=35):
        import soundfile as sf
        audio, sample_rate = self.generate(text, seed)
        dest = Path(output).expanduser().resolve()
        dest.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(dest), audio, sample_rate)
        return {"output": str(dest), "sample_rate": sample_rate, "duration_seconds": len(audio) / sample_rate,
                "voice": str(self.voice_dir), "reference_used": bool(self.reference_wav),
                "engine": "voxcpm2", "streaming": False}
