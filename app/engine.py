
import os

import mlx_whisper
from pyannote.audio import Pipeline



def format_time(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def get_overlap(w_start, w_end, s_start, s_end):
    return min(w_end, s_end) - max(w_start, s_start)

def assign_speaker(w_start, w_end, diarization_segments):
    max_overlap = 0
    best_speaker = None
    for segment in diarization_segments:
        overlap = get_overlap(w_start, w_end, segment['start'], segment['end'])
        if overlap>max_overlap:
            max_overlap=overlap
            best_speaker=segment['speaker']
    if best_speaker == None:
        best_speaker = "UNKNOWN"
    
    return best_speaker


# Diarize
hf_token = os.environ["HF_TOKEN"]
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token,
)


def make_transcript(audio_path: str, num_speakers: int = 2) -> str:
    result = mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
    )
    whisper_segments = result["segments"]
    diarization = pipeline(audio_path, num_speakers=num_speakers)
    diarization_segments = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    lines = []
    for segment in whisper_segments:
        speaker = assign_speaker(
            segment["start"], segment["end"], diarization_segments
        )
        timestamp = format_time(segment["start"])
        lines.append(f"[{timestamp}] {speaker}: {segment['text']}")
    return "\n".join(lines)