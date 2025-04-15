#!/usr/bin/env python3

"""
Comprehensive test script for the pronunciation model.
This script records audio, processes it, and runs the pronunciation model.
"""

import os
import sys
import torch
import numpy as np

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import sounddevice as sd
    import scipy.io.wavfile as wav
    from app.services.pronunciation.audio_utils import load_audio
    from app.services.pronunciation.feature_extractor import extract_mfcc
    from app.services.pronunciation.model import PronunciationClassifier
    
    # Define the record function (similar to mic_recorder.py but included here for simplicity)
    def record_audio(filename="temp_audio.wav", duration=5, fs=16000):
        print("🎤 Recording audio for", duration, "seconds...")
        print("Please speak now.")
        audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()
        wav.write(filename, fs, audio)
        print(f"✅ Audio saved to {filename}")
        return filename
    
    def predict_pronunciation(audio_file):
        print("Loading audio file...")
        audio = load_audio(audio_file)
        
        print("Extracting MFCC features...")
        mfcc = extract_mfcc(audio)
        mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)  # [1, time, features]
        
        print("Initializing pronunciation model...")
        model = PronunciationClassifier()
        
        # Check if the model file exists
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "pronunciation_model.pt")
        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found at {model_path}.")
            print("Creating a dummy model for testing purposes.")
            
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save a dummy model
            torch.save(model.state_dict(), model_path)
        
        # Load the model
        print("Loading model weights...")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        # Make prediction
        print("Making prediction...")
        with torch.no_grad():
            score = model(mfcc_tensor).item()
        
        return {
            "score": round(score * 100, 2),
            "status": "Correct" if score > 0.5 else "Incorrect"
        }
    
    def main():
        print("\n📢 Pronunciation Model Test\n" + "-" * 30)
        
        # Ask user if they want to record new audio
        record_new = input("Do you want to record new audio? (y/n): ").lower() == 'y'
        
        if record_new:
            # Ask for duration
            try:
                duration = float(input("How many seconds to record? (default: 5): ") or "5")
            except ValueError:
                duration = 5
                
            # Record audio
            audio_file = record_audio(duration=duration)
        else:
            # Use existing file
            audio_file = "temp_audio.wav"
            if not os.path.exists(audio_file):
                print(f"❌ Error: Audio file {audio_file} not found.")
                print("Please run the script again and record new audio.")
                return
        
        # Predict pronunciation
        try:
            result = predict_pronunciation(audio_file)
            
            # Print results
            print("\n🔍 Pronunciation Analysis Results:")
            print(f"Score: {result['score']}%")
            print(f"Status: {result['status']}")
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")
    
    if __name__ == "__main__":
        main()
        
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please install all required dependencies with pip install -r requirements.txt") 