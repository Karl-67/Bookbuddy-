# backend/app/main.py

from fastapi import FastAPI
from app.routes import feedback, query, analyze
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(feedback.router, prefix="/feedback", tags=["feedback"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(analyze.router, prefix="/analyze", tags=["analyze"])

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",  # Tells uvicorn where the app instance is
        host="0.0.0.0",   # Allows external access if deployed
        port=8000,
        reload=True       # Enables hot-reload during development
    )
