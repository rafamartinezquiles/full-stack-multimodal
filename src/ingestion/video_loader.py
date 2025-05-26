import whisper
import moviepy.editor as mp

model = whisper.load_model("base")

def extract_audio_text_from_video(video_path: str) -> str:
    # Extract audio from video
    video = mp.VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    video.audio.write_audiofile(audio_path, logger=None)

    # Transcribe with Whisper
    result = model.transcribe(audio_path)
    return result["text"].strip()
