#!/usr/bin/env python3

"""
Script to run speech-to-text functionality of BookBuddy.
This will record from your microphone until silence is detected,
then transcribe the audio using Google Cloud Speech-to-Text.
"""

import os
import sys

# Add the parent directory to the path so we can import the app modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.speech_to_text import live_transcribe, record_from_microphone, transcribe_audio

def main():
    print("\n🎙️ BookBuddy Speech-to-Text Test\n" + "-" * 35)
    
    # Ask user which mode to use
    print("1. Live transcription (record until silence)")
    print("2. Record for specific duration")
    print("3. Transcribe existing audio file")
    
    try:
        choice = int(input("\nSelect an option (1-3): ") or "1")
    except ValueError:
        choice = 1
    
    if choice == 1:
        print("\nStarting live transcription...")
        text = live_transcribe()
        print("\n📝 Transcription Result:")
        print(text)
    
    elif choice == 2:
        try:
            duration = float(input("\nHow many seconds to record? (default: 5): ") or "5")
            max_seconds = float(input("Maximum recording time? (default: 30): ") or "30")
            silence_threshold = int(input("Silence threshold (lower = more sensitive, default: 800): ") or "800")
            silence_duration = float(input("Silence duration to end recording (seconds, default: 2.5): ") or "2.5")
        except ValueError:
            duration = 5
            max_seconds = 30
            silence_threshold = 800
            silence_duration = 2.5
        
        print("\nRecording...")
        temp_file = "temp_audio.wav"
        record_from_microphone(
            filename=temp_file,
            silence_threshold=silence_threshold,
            silence_duration=silence_duration,
            max_record_seconds=max_seconds
        )
        
        print("\nTranscribing audio...")
        text = transcribe_audio(temp_file)
        
        print("\n📝 Transcription Result:")
        print(text)
    
    elif choice == 3:
        audio_file = input("\nEnter the path to the audio file: ")
        if not os.path.exists(audio_file):
            print(f"❌ Error: File {audio_file} not found.")
            return
        
        print("\nTranscribing audio...")
        text = transcribe_audio(audio_file)
        
        print("\n📝 Transcription Result:")
        print(text)
    
    else:
        print("Invalid option selected.")

if __name__ == "__main__":
    main() 