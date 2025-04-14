#!/usr/bin/env python3

"""
Simple script to test audio loading and feature extraction components.
"""

import os
import sys
import numpy as np

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.pronunciation.audio_utils import load_audio
from app.services.pronunciation.feature_extractor import extract_mfcc

def main():
    print("Testing audio processing components")
    
    # Use the temp_audio.wav file if it exists
    temp_file = "temp_audio.wav"
    if os.path.exists(temp_file):
        print(f"Found existing audio file: {temp_file}")
        
        # Load the audio
        print("Loading audio...")
        audio = load_audio(temp_file)
        print(f"Audio loaded, shape: {audio.shape if hasattr(audio, 'shape') else 'Unknown'}")
        
        # Extract MFCC features
        print("Extracting MFCC features...")
        mfcc = extract_mfcc(audio)
        print(f"MFCC features extracted, shape: {mfcc.shape if hasattr(mfcc, 'shape') else 'Unknown'}")
        
        print("\nFeature statistics:")
        if hasattr(mfcc, 'shape'):
            print(f"Min: {np.min(mfcc)}")
            print(f"Max: {np.max(mfcc)}")
            print(f"Mean: {np.mean(mfcc)}")
            print(f"Std: {np.std(mfcc)}")
    else:
        print(f"Error: Audio file '{temp_file}' not found.")

if __name__ == "__main__":
    main() 