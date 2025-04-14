import os
import wave
import sounddevice as sd
import numpy as np
from google.cloud import speech
import time

# Set Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    os.path.dirname(__file__), "..", "credentials", "google_speech_key.json"
)

def record_until_silence_v2(
    filename="temp_audio.wav",
    rate=16000,
    silence_threshold=100,  # Very low threshold for silence detection
    pause_threshold=1.0,    # Time of silence before stopping
    max_duration=30,        # Maximum recording duration
    energy_buffer_size=10   # How many frames to check for energy level
):
    """
    Records audio until silence using a different approach based on energy levels.
    
    Args:
        filename (str): Output file path
        rate (int): Sample rate
        silence_threshold (int): Energy level threshold for silence
        pause_threshold (float): How long silence must persist to stop recording (seconds)
        max_duration (int): Maximum recording duration (seconds)
        energy_buffer_size (int): How many previous frames to check for energy
    """
    print("🎙️ Recording... (speak now, will stop automatically after silence)")
    
    # Calculate parameters
    chunk_size = int(rate * 0.1)  # 100ms chunks for responsive detection
    max_chunks = int(max_duration * 10)  # 10 chunks per second
    pause_chunks = int(pause_threshold * 10)  # How many chunks of silence before stopping
    
    all_audio = []
    energy_history = []
    silent_chunk_count = 0
    has_speech = False
    recording_started = time.time()
    
    # Create an input stream
    with sd.InputStream(samplerate=rate, channels=1, dtype='int16', blocksize=chunk_size) as stream:
        for i in range(max_chunks):
            # Read audio chunk
            audio_chunk, _ = stream.read(chunk_size)
            
            # Convert to numpy array for processing
            audio_chunk = audio_chunk.reshape(-1)
            all_audio.append(audio_chunk)
            
            # Calculate energy (squared sum is faster than RMS and works for this purpose)
            energy = np.sum(np.square(audio_chunk.astype(np.float32))) / len(audio_chunk)
            energy_history.append(energy)
            
            # Keep the history buffer at the right size
            if len(energy_history) > energy_buffer_size:
                energy_history.pop(0)
            
            # Calculate the average energy from the buffer
            avg_energy = sum(energy_history) / len(energy_history)
            
            # Debug output - uncomment to see values
            # print(f"Energy: {energy:.1f}, Avg: {avg_energy:.1f}, Silent chunks: {silent_chunk_count}/{pause_chunks}")
            
            # Detect if we have speech (needed before we can end on silence)
            if avg_energy > silence_threshold * 2:  # Higher threshold for speech detection
                has_speech = True
                silent_chunk_count = 0
            elif has_speech and avg_energy < silence_threshold:
                silent_chunk_count += 1
            else:
                silent_chunk_count = 0
            
            # Check if we should stop recording
            if has_speech and silent_chunk_count >= pause_chunks:
                print("✅ Silence detected, stopping recording.")
                break
            
            # Safety check for max duration
            if time.time() - recording_started > max_duration:
                print("⚠️ Maximum recording duration reached.")
                break
    
    # Make sure we have audio
    if not all_audio:
        print("⚠️ No audio recorded!")
        return
    
    # Combine all chunks
    recording = np.concatenate(all_audio)
    
    # Remove trailing silence
    if silent_chunk_count > 0 and len(all_audio) > silent_chunk_count:
        recording = np.concatenate(all_audio[:-silent_chunk_count])
    
    # Save to WAV file
    duration = len(recording) / rate
    print(f"✅ Finished recording ({duration:.1f} seconds). Saving to {filename}")
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes per sample for int16
        wf.setframerate(rate)
        wf.writeframes(recording.tobytes())

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes speech from an audio file using Google Cloud Speech-to-Text API.

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
        language_code="en-US",
        enable_automatic_punctuation=True
    )

    response = client.recognize(config=config, audio=audio)

    if not response.results:
        print("⚠️ No transcription returned. Possible silence or bad audio.")
        return ""

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript + " "

    return transcript.strip()

def live_transcribe():
    temp_filename = "temp_audio.wav"
    record_until_silence_v2(filename=temp_filename)
    transcript = transcribe_audio(temp_filename)
    return transcript

if __name__ == "__main__":
    print("Starting live transcription...")
    text = live_transcribe()
    print("Transcript:", text)