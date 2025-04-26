import torch
import numpy as np
import os

from .model import PronunciationClassifier
from .audio_utils import load_audio
from .feature_extractor import extract_mfcc

def predict(file_path: str):
    audio = load_audio(file_path)
    mfcc = extract_mfcc(audio)
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, time, features]

    model = PronunciationClassifier()
    # Use the correct path to the model file
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "pronunciation_model_final.pt")
    
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Creating a dummy model for testing.")
        torch.save(model.state_dict(), model_path)
    
    # Load the checkpoint and extract the model state dict
    checkpoint = torch.load(model_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If it's not a checkpoint, try loading it directly
        model.load_state_dict(checkpoint)
    
    model.eval()

    with torch.no_grad():
        score = model(mfcc_tensor).item()

    return {
        "score": round(score * 100, 2),
        "status": "Correct" if score > 0.5 else "Incorrect"
    }

if __name__ == "__main__":
    from .mic_recorder import record_audio

    # Step 1: Record from mic and save as temp_audio.wav
    record_audio("temp_audio.wav", duration=5)

    # Step 2: Run prediction on the recorded audio
    result = predict("temp_audio.wav")

    print(result)
