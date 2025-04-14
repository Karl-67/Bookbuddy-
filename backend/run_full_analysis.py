#!/usr/bin/env python3

"""
Combined script for BookBuddy's full speech analysis.
This script handles both speech-to-text transcription and pronunciation quality assessment.
"""

import os
import sys

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.speech_to_text import record_from_microphone, transcribe_audio
from app.services.pronunciation.predict import predict

def main():
    print("\n🎙️ BookBuddy Full Speech Analysis\n" + "-" * 40)
    
    # Record audio
    audio_file = "temp_audio.wav"
    print("\n1️⃣ Recording Audio")
    print("------------------")
    print("Please speak into your microphone. Recording will stop after silence is detected.")
    
    # Record with silence detection
    record_from_microphone(
        filename=audio_file,
        silence_threshold=800,
        silence_duration=2.5,
        max_record_seconds=30
    )
    
    # Transcribe audio
    print("\n2️⃣ Speech-to-Text Analysis")
    print("-------------------------")
    print("Transcribing your speech...")
    transcript = transcribe_audio(audio_file)
    
    print("\n📝 Transcription Result:")
    print(transcript)
    
    # Analyze pronunciation
    print("\n3️⃣ Pronunciation Analysis")
    print("------------------------")
    print("Analyzing pronunciation quality...")
    pronunciation_result = predict(audio_file)
    
    print("\n📊 Pronunciation Score:")
    print(f"Score: {pronunciation_result['score']}%")
    print(f"Status: {pronunciation_result['status']}")
    
    # Combined results
    print("\n📋 Summary Report:")
    print("-----------------")
    print(f"✓ Transcript: {transcript}")
    print(f"✓ Pronunciation Score: {pronunciation_result['score']}%")
    print(f"✓ Pronunciation Status: {pronunciation_result['status']}")
    
if __name__ == "__main__":
    main() 