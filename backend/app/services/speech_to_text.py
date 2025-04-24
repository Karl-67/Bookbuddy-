import os
import wave
#import pyaudio
import audioop  # For calculating RMS of audio samples
from dotenv import load_dotenv
from google.cloud import speech

# ✅ Load the .env file
load_dotenv()

# ✅ Set the required env variable for Google SDK from the .env
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


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
    client = speech.SpeechClient()

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


def live_transcribe():
    temp_filename = "temp_audio.wav"
    record_from_microphone(filename=temp_filename)
    transcript = transcribe_audio(temp_filename)

    print(f"🎧 Transcript: {transcript}")  # 👈 add this
    if not transcript.strip():
        return "❌ No speech detected."
    return transcript



# Run it for testing
if __name__ == "__main__":
    print("Starting live transcription...")
    text = live_transcribe()
    print("Transcript:", text)
