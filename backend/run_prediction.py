#!/usr/bin/env python3

"""
Script to test the pronunciation prediction functionality.
"""

import os
import sys

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.pronunciation.mic_recorder import record_audio
from app.services.pronunciation.predict import predict

def main():
    print("🎤 Testing pronunciation prediction")
    
    # Record audio
    temp_file = "temp_audio.wav"
    print("Recording 5 seconds of audio...")
    record_audio(temp_file, duration=5)
    
    # Make prediction
    print("Analyzing pronunciation...")
    result = predict(temp_file)
    
    print("\nResults:")
    print(f"Score: {result['score']}%")
    print(f"Status: {result['status']}")
    
    return result

if __name__ == "__main__":
    main() 