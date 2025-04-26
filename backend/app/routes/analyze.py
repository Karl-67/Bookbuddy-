from fastapi import APIRouter
from app.services.speech_to_text import live_transcribe
from app.services.text_simplifier import simplify_text
from app.services.pronunciation.predict import predict
import os

router = APIRouter()

@router.post("/")
async def analyze_speech():
    try:
        # Step 1: Record audio and get transcript
        temp_filename = "temp_audio.wav"
        transcript = live_transcribe()
        
        if transcript.startswith("❌"):
            return {"error": transcript}
            
        # Step 2: Analyze pronunciation using both audio and transcript
        pronunciation_result = predict(temp_filename, transcript)
        
        # Step 3: Simplify the text
        simplified = simplify_text(transcript)
        
        return {
            "transcript": transcript,
            "simplified": simplified,
            "pronunciation_score": pronunciation_result["score"],
            "pronunciation_status": pronunciation_result["status"]
        }
    except Exception as e:
        return {"error": str(e)} 