#!/usr/bin/env python3

"""
Script to run the pronunciation prediction functionality.
"""

import os
import sys

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.pronunciation.predict import predict
from app.services.pronunciation.mic_recorder import record_audio

def main():
    print("\n🔊 Pronunciation Prediction Test\n" + "-" * 35)
    
    # Check if temp_audio.wav exists
    audio_file = "temp_audio.wav"
    if not os.path.exists(audio_file):
        print(f"Audio file '{audio_file}' not found.")
        record_new = input("Would you like to record new audio? (y/n): ").lower() == 'y'
        
        if record_new:
            try:
                duration = float(input("How many seconds to record? (default: 5): ") or "5")
            except ValueError:
                duration = 5
            
            print("\nRecording...")
            record_audio(audio_file, duration=duration)
        else:
            print("Cannot proceed without an audio file. Exiting.")
            return
    
    # Run prediction
    print("\nAnalyzing pronunciation...")
    result = predict(audio_file)
    
    # Print results
    print("\n📊 Pronunciation Analysis Results:")
    print(f"Score: {result['score']}%")
    print(f"Status: {result['status']}")

if __name__ == "__main__":
    main() 