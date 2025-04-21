from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from speech_to_text import live_transcribe  # import from your existing file
from text_explainer import simplify_text  # Your OpenAI code

import uvicorn

app = FastAPI()

# CORS for frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
def transcribe_and_simplify():
    transcript = live_transcribe()
    simplified = simplify_text(transcript)
    return {
        "transcript": transcript,
        "simplified": simplified
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
