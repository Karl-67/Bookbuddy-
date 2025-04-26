import torch
import numpy as np
import os
import logging

from .model import PronunciationClassifier
from .audio_utils import load_audio
from .feature_extractor import extract_mfcc

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict(file_path: str):
    try:
        logger.info(f"Starting prediction for file: {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
            
        # Load and process audio
        logger.info("Loading audio file...")
        audio = load_audio(file_path)
        logger.info(f"Audio loaded, shape: {audio.shape}, min: {audio.min()}, max: {audio.max()}")
        
        # Extract features
        logger.info("Extracting MFCC features...")
        mfcc = extract_mfcc(audio)
        logger.info(f"MFCC features extracted, shape: {mfcc.shape}, min: {mfcc.min()}, max: {mfcc.max()}")
        
        # Convert to tensor and normalize
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32)
        # Normalize the features
        mfcc_tensor = (mfcc_tensor - mfcc_tensor.mean()) / (mfcc_tensor.std() + 1e-8)
        mfcc_tensor = mfcc_tensor.unsqueeze(0)  # Add batch dimension
        logger.info(f"Converted to tensor, shape: {mfcc_tensor.shape}, min: {mfcc_tensor.min()}, max: {mfcc_tensor.max()}")

        # Initialize model
        logger.info("Initializing model...")
        model = PronunciationClassifier()
        
        # Use the best model file
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                 "models", "pronunciation_model_best.pt")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        
        # Load the model
        logger.info(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
            
        # Load state dict with strict=False to handle mismatches
        model.load_state_dict(state_dict, strict=False)
        logger.info("Model loaded successfully")
        
        model.eval()
        logger.info("Model set to eval mode")

        # Run prediction
        logger.info("Running prediction...")
        with torch.no_grad():
            # Get raw output before sigmoid
            raw_output = model.fc(model.lstm(mfcc_tensor)[0].mean(dim=1))
            logger.info(f"Raw model output before sigmoid: {raw_output.item()}")
            
            # Get final score
            score = model(mfcc_tensor).item()
            logger.info(f"Final score after sigmoid: {score}")

        # Use balanced thresholds with adjusted ranges
        if score > 0.3:  # Good pronunciation (lowered from 0.4)
            status = "Good"
            # Scale up good scores more aggressively
            scaled_score = 50 + (score - 0.3) * 200  # More generous scaling for good pronunciations
        elif score > 0.15:  # Fair pronunciation (lowered from 0.2)
            status = "Fair"
            # Scale fair scores more generously
            scaled_score = 30 + (score - 0.15) * 200
        else:  # Needs improvement
            status = "Needs Improvement"
            # Scale lower scores more generously
            scaled_score = score * 200  # More generous scaling for lower scores

        # Ensure score is within 0-100 range
        final_score = min(100, max(0, scaled_score))
        logger.info(f"Raw score: {score:.4f}, Scaled score: {final_score:.2f}, Status: {status}")

        return {
            "score": round(final_score, 2),
            "status": status
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}", exc_info=True)
        return {
            "score": 0,
            "status": "Error",
            "error": str(e)
        }

if __name__ == "__main__":
    from .mic_recorder import record_audio

    # Step 1: Record from mic and save as temp_audio.wav
    record_audio("temp_audio.wav", duration=5)

    # Step 2: Run prediction on the recorded audio
    result = predict("temp_audio.wav")

    print(result)
