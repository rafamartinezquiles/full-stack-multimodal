import whisper

model = whisper.load_model("base")

def extract_text_from_audio(audio_path: str) -> str:
    result = model.transcribe(audio_path)
    return result["text"].strip()
