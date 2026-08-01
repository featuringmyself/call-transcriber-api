import shutil
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException


from app.engine import make_transcript

app = FastAPI()

@app.post("/")
async def root(file: UploadFile = File(...)):
    suffix = Path(file.filename or "audio").suffix or ".mp3"

    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / f"upload{suffix}"
        try:
            with dest.open("wb") as out:
                shutil.copyfileobj(file.file, out)
        except Exception:
            raise HTTPException(status_code = 400, detail="failed to save")
        
        transcript = make_transcript(str(dest))

    return {"transcript": transcript}