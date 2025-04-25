from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from speech_to_text import live_transcribe
from text_explainer import simplify_text
from text_to_speech import synthesize_speech
from dotenv import load_dotenv
import uvicorn
import time
import os

# Load environment variables
load_dotenv()

app = FastAPI()

# ✅ CORS config
origins = [
    "http://localhost:5173",  # Vite default port
    "http://localhost:3000",  # React default port
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.post("/transcribe")
async def transcribe_and_simplify():
    try:
        transcript = live_transcribe()
        simplified = simplify_text(transcript)
        return {
            "transcript": transcript,
            "simplified": simplified
        }
    except Exception as e:
        print(f"Error in transcribe endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to transcribe: {str(e)}"}
        )

@app.post("/tts")
async def tts(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        
        if not text.strip():
            return JSONResponse(
                status_code=400,
                content={"error": "No text provided for speech synthesis"}
            )
            
        # Generate a unique filename for each request
        output_path = f"tts_output_{int(time.time())}.mp3"
        audio_path = synthesize_speech(text, output_path)

        if audio_path.startswith("❌"):
            return JSONResponse(
                status_code=500,
                content={"error": audio_path}
            )

        # Check if the file exists and is readable
        if not os.path.exists(audio_path):
            return JSONResponse(
                status_code=500,
                content={"error": "Generated audio file not found"}
            )

        # Return the audio file
        return FileResponse(
            audio_path,
            media_type="audio/mpeg",
            filename="response.mp3",
            background=None  # This ensures the file is deleted after sending
        )
    except Exception as e:
        print(f"Error in TTS endpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process TTS request: {str(e)}"}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
