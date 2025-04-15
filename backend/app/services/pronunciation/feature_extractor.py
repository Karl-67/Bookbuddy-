import numpy as np
import librosa

def extract_mfcc(audio: np.ndarray, sr: int = 16000, n_mfcc: int = 13):
    """
    Extracts MFCC features from raw audio.
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T  # Shape: (time, features)
