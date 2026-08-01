from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

from app.engine import make_transcript

app = FastAPI()


class TranscribeRequest(BaseModel):
    path: str

@app.get("/")
def root(body: TranscribeRequest):
    audio = Path(body.path)
    
    if not audio.is_file():
        raise HTTPException(status_code=400, detail="File not found")
    transcript = make_transcript(audio)
    return {"transcript": transcript}