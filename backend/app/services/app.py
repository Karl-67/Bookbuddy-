from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from speech_to_text import live_transcribe  # import from your existing file
import uvicorn

app = FastAPI()

# Allow frontend requests (React runs on port 5173 typically)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5173"] for stricter access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe")
def transcribe():
    transcript = live_transcribe()
    return {"transcript": transcript}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
