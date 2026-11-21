from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from .validator import inspect_voice

app = FastAPI(title="VoxCPM Real-Time Voice Engine", version="0.1.0")

class ValidationRequest(BaseModel):
    folder: str

@app.get("/health")
def health():
    return {"status": "ok", "phase": 1, "inference_ready": False}

@app.post("/api/voices/validate")
def validate(request: ValidationRequest):
    # Local-only scaffold. Do not expose this filesystem endpoint to untrusted networks.
    result = inspect_voice(Path(request.folder))
    if not result["compatible"]:
        raise HTTPException(status_code=422, detail=result)
    return result

@app.get("/api/capabilities")
def capabilities():
    return {"validation": True, "synthesis": False, "streaming": False, "runtime_verified": False}
