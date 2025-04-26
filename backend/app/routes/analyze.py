from fastapi import APIRouter
from app.services.speech_to_text import live_transcribe
from app.services.text_simplifier import simplify_text
from app.services.pronunciation_analyzer import analyze_pronunciation
import os

router = APIRouter()

@router.post("/")
async def analyze_speech():
    try:
        # Get the transcript from speech-to-text
        transcript = live_transcribe()
        
        if transcript.startswith("❌"):
            return {"error": transcript}
            
        # Simplify the text
        simplified = simplify_text(transcript)
        
        # Analyze pronunciation
        pronunciation_result = analyze_pronunciation(transcript)
        
        return {
            "transcript": transcript,
            "simplified": simplified,
            "pronunciation_score": pronunciation_result.get("score", 0),
            "pronunciation_status": pronunciation_result.get("status", "unknown")
        }
    except Exception as e:
        return {"error": str(e)} 