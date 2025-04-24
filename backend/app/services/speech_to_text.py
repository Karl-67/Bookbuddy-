import os
import wave
#import pyaudio
import audioop  # For calculating RMS of audio samples
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    os.path.dirname(__file__), "..", "credentials", "google_speech_key.json"
)

from google.cloud import speech

def record_from_microphone(
    filename="temp_audio.wav",
    silence_threshold=800,     # Adjust this threshold as needed
    silence_duration=2.5,        # Seconds of silence required to stop recording
    max_record_seconds=10        # Maximum recording length as a safety net
):
    """
    Records audio from the default microphone until silence is detected.

    Args:
        filename (str): File to save the recording.
        silence_threshold (int): RMS threshold to consider as silence.
        silence_duration (float): Duration in seconds of consecutive silence to stop recording.
        max_record_seconds (int): Safety limit for recording duration.

    Returns:
        None: The audio is saved to the specified filename.
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

        # Calculate RMS value to check for silence
        rms = audioop.rms(data, 2)  # 2 bytes per sample for paInt16
        #print(f"RMS: {rms}, Silent Chunks: {silent_chunks}/{required_silent_chunks}")
        if rms < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0
        
        # If enough consecutive silent chunks detected, break out of the loop
        if silent_chunks >= required_silent_chunks:
            print("✅ Detected silence. Stopping recording.")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded frames as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes speech from an audio file using the Google Cloud Speech-to-Text API.
    
    Args:
        audio_path (str): Path to the audio file.
    
    Returns:
        str: The transcribed text.
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
    """
    Records from the microphone (with silence detection) and transcribes the audio.
    
    Returns:
        str: The transcribed text.
    """
    temp_filename = "temp_audio.wav"
    record_from_microphone(filename=temp_filename)
    transcript = transcribe_audio(temp_filename)
    # Optionally, remove the temporary file after transcription.
    # os.remove(temp_filename)
    return transcript

# Example usage (for testing):
if __name__ == "__main__":
    print("Starting live transcription...")
    text = live_transcribe()
    print("Transcript:", text)
