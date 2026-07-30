from moviepy import AudioFileClip


def mp4_to_mp3(video_path: str, audio_path: str) -> None:
    with AudioFileClip(video_path) as audio:
        audio.write_audiofile(audio_path, bitrate="320k")


if __name__ == "__main__":
    mp4_to_mp3("sample.mp4", "sample.mp3")
