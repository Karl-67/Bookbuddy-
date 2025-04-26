import wave
import pyaudio
import audioop  # For calculating RMS of audio samples
from dotenv import load_dotenv
from google.cloud import speech
from google.auth.exceptions import DefaultCredentialsError
import concurrent.futures

# ✅ Load the .env file
load_dotenv()

import os

# Global client for reuse
speech_client = None

def get_credentials():
    """Get Google Cloud credentials path from environment or .env file"""
    # First try the environment variable
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    
    # If not set, try the .env file
    if not credentials_path:
        credentials_path = os.getenv("Text_to_Speech")
        if credentials_path:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    
    if not credentials_path:
        raise ValueError("❌ Missing Google Cloud credentials. Please set GOOGLE_APPLICATION_CREDENTIALS or Text_to_Speech in .env")
    
    if not os.path.exists(credentials_path):
        raise FileNotFoundError(f"❌ Credentials file not found at: {credentials_path}")
    
    print(f"🔍 Using credentials from: {credentials_path}")
    return credentials_path

def get_speech_client():
    """Get and cache the speech client"""
    global speech_client
    if speech_client is None:
        get_credentials()
        speech_client = speech.SpeechClient()
    return speech_client

def record_from_microphone(
    filename="temp_audio.wav",
    silence_threshold=600,     # Adjust this threshold as needed
    silence_duration=2.5,      # Seconds of silence required to stop recording
    max_record_seconds=15      # Maximum recording length as a safety net
):
    """
    Records audio from the default microphone until silence is detected.
    """
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 16000  # Sample rate for Google STT

    p = pyaudio.PyAudio()
    stream = p.open(format=format,
                    channels=channels,
                    rate=rate,
                    input=True,
                    frames_per_buffer=chunk)

    print("🎙️ Recording... Speak now. (Recording will stop after silence)")

    frames = []
    silent_chunks = 0
    required_silent_chunks = int((silence_duration * rate) / chunk)
    total_chunks = int((max_record_seconds * rate) / chunk)
    
    for i in range(total_chunks):
        data = stream.read(chunk)
        frames.append(data)

        rms = audioop.rms(data, 2)  # 2 bytes per sample for paInt16
        if rms < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0
        
        if silent_chunks >= required_silent_chunks:
            print("✅ Detected silence. Stopping recording.")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes speech from an audio file using Google STT API.
    """
    global speech_client  # Move this to the top of the function
    
    try:
        # Use the cached client instead of creating a new one
        client = get_speech_client()

        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US"
        )

        response = client.recognize(config=config, audio=audio)

        transcript = ""
        for result in response.results:
            transcript += result.alternatives[0].transcript + " "

        return transcript.strip()
    except DefaultCredentialsError as e:
        print(f"❌ Google Cloud credentials error: {e}")
        # Clear client to force re-initialization on next try
        speech_client = None  # Already declared global at the top
        raise
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        # Reset client on any error to force re-initialization
        speech_client = None  # Already declared global at the top
        raise


def analyze_and_transcribe():
    """
    Records audio, then analyzes pronunciation and transcribes in parallel.
    Returns both the pronunciation score and transcript.
    """
    try:
        temp_filename = "temp_audio.wav"
        
        # Step 1: Record audio
        record_from_microphone(filename=temp_filename)
        
        # Step 2: Run pronunciation analysis and transcription in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Import predict here to avoid circular imports
            from app.services.pronunciation.predict import predict
            
            # Submit both tasks
            pronunciation_future = executor.submit(predict, temp_filename)
            transcript_future = executor.submit(transcribe_audio, temp_filename)
            
            # Wait for both to complete
            pronunciation_result = pronunciation_future.result()
            transcript = transcript_future.result()
        
        if not transcript.strip():
            return {
                "error": "❌ No speech detected.",
                "pronunciation_score": 0,
                "pronunciation_status": "unknown"
            }
            
        return {
            "transcript": transcript,
            "pronunciation_score": pronunciation_result["score"],
            "pronunciation_status": pronunciation_result["status"]
        }
        
    except Exception as e:
        print(f"❌ Error in analysis and transcription: {e}")
        return {
            "error": str(e),
            "pronunciation_score": 0,
            "pronunciation_status": "unknown"
        }


def live_transcribe():
    try:
        temp_filename = "temp_audio.wav"
        record_from_microphone(filename=temp_filename)
        transcript = transcribe_audio(temp_filename)

        print(f"🎧 Transcript: {transcript}")
        if not transcript.strip():
            return "❌ No speech detected."
        return transcript
    except Exception as e:
        print(f"❌ Error in live transcription: {e}")
        return f"❌ Error: {str(e)}"


# Run it for testing
if __name__ == "__main__":
    print("Starting live transcription...")
    text = live_transcribe()
    print("Transcript:", text)