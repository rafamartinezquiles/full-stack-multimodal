# Import the Whisper library for automatic speech recognition (ASR)
import whisper

# Load the pre-trained Whisper model (using the "base" size variant)
# You can replace "base" with "small", "medium", or "large" for different tradeoffs
model = whisper.load_model("base")

# Define a function to extract transcribed text from an audio file
def extract_text_from_audio(audio_path: str) -> str:
    # Use the Whisper model to transcribe the audio at the given path
    result = model.transcribe(audio_path)
    
    # Return only the transcribed text, removing any leading/trailing whitespace
    return result["text"].strip()
