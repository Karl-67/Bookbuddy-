import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

def record_audio(filename="temp_audio.wav", duration=5, fs=16000):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, fs, audio)
    print(f"✅ Saved to {filename}")
