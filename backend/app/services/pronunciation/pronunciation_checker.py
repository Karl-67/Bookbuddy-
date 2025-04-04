import torch
from services.pronunciation.audio_utils import load_audio
from services.pronunciation.feature_extractor import extract_mfcc
from services.pronunciation.model import PronunciationClassifier

def check_pronunciation(audio_path: str) -> dict:
    """
    Full pipeline to check pronunciation from an audio file.
    Returns a confidence score and classification.
    """
    # Step 1: Load and process audio
    audio = load_audio(audio_path)
    mfcc = extract_mfcc(audio)

    # Step 2: Prepare input tensor
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, time, features]

    # Step 3: Load model
    model = PronunciationClassifier()
    model.load_state_dict(torch.load("models/pronunciation_model.pt"))
    model.eval()

    # Step 4: Run inference
    with torch.no_grad():
        score = model(mfcc_tensor).item()

    # Step 5: Return structured output
    return {
        "confidence": round(score * 100, 2),
        "status": "Correct" if score > 0.5 else "Incorrect"
    }
