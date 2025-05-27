# Import the Whisper speech recognition library
import whisper

# Import moviepy for handling video and audio extraction
import moviepy.editor as mp

# Load the Whisper model (base version)
# You can use other sizes: "tiny", "small", "medium", "large" depending on accuracy/performance needs
model = whisper.load_model("base")

# Define a function to extract and transcribe speech from a video file
def extract_audio_text_from_video(video_path: str) -> str:
    # Load the video file using moviepy
    video = mp.VideoFileClip(video_path)

    # Define a temporary path for storing the extracted audio
    audio_path = "temp_audio.wav"

    # Extract the audio track from the video and save it as a WAV file
    video.audio.write_audiofile(audio_path, logger=None)  

    # Transcribe the extracted audio using Whisper
    result = model.transcribe(audio_path)

    # Return the transcribed text with leading/trailing whitespace removed
    return result["text"].strip()
