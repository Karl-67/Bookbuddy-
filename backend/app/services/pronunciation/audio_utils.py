import librosa

def load_audio(file_path: str, sr: int = 16000):
    """
    Loads audio from a file path and resamples to the target sampling rate.
    """
    audio, _ = librosa.load(file_path, sr=sr)
    return audio
