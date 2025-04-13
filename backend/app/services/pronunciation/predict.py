import torch
from .dataset import PronunciationClassifier
from .audio_utils import load_audio
from .feature_extractor import extract_mfcc
import numpy as np

def predict(file_path: str):
    audio = load_audio(file_path)
    mfcc = extract_mfcc(audio)
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, time, features]

    model = PronunciationClassifier()
    model.load_state_dict(torch.load("models/pronunciation_model.pt"))
    model.eval()

    with torch.no_grad():
        score = model(mfcc_tensor).item()

    return {
        "score": round(score * 100, 2),
        "status": "Correct" if score > 0.5 else "Incorrect"
    }
#if __name__ == "__main__":
    from services.pronunciation.mic_recorder import record_audio

    # Step 1: Record from mic and save as temp_audio.wav
    record_audio("temp_audio.wav", duration=5)

    # Step 2: Run prediction on the recorded audio
    result = predict("temp_audio.wav")

    print(result)
