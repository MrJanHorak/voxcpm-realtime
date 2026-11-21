"""Local-only API. Run one worker to keep exactly one model in GPU memory."""
import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from .manager import ModelManager
from .validator import inspect_voice

app = FastAPI(title="VoxCPM Real-Time Voice Engine", version="0.3.0")
manager = ModelManager(os.getenv("VOXCPM_MODEL_DIR", "models/VoxCPM2"),
                       os.getenv("VOXCPM_VOICE_DIR", "voices/step150"),
                       os.getenv("VOXCPM_REFERENCE_WAV") or None)

class ValidationRequest(BaseModel):
    folder: str

class SynthesisRequest(BaseModel):
    text: str = Field(min_length=1)
    seed: int | None = 35

@app.get("/health")
def health():
    return {"status": "ok", "phase": 3, "model_loaded": manager.loaded, "warmed": manager.warmed, "inference_ready": manager.loaded}

@app.post("/api/voices/validate")
def validate(request: ValidationRequest):
    result = inspect_voice(Path(request.folder))
    if not result["compatible"]:
        raise HTTPException(status_code=422, detail=result)
    return result

@app.get("/api/capabilities")
def capabilities():
    return {"validation": True, "synthesis": True, "streaming": False,
            "model_loaded": manager.loaded, "reference_cache": True}

@app.post("/api/warmup")
def warmup():
    try:
        manager.warmup()
        return {"status": "ready", "model_load_seconds": manager.load_seconds}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.post("/api/synthesize")
def synthesize(request: SynthesisRequest):
    import io
    import soundfile as sf
    try:
        audio, rate = manager.generate(request.text, request.seed)
        buffer = io.BytesIO()
        sf.write(buffer, audio, rate, format="WAV")
        return Response(content=buffer.getvalue(), media_type="audio/wav")
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
