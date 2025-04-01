# backend/app/main.py

from app import app

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",  # Tells uvicorn where the app instance is
        host="0.0.0.0",   # Allows external access if deployed
        port=8000,
        reload=True       # Enables hot-reload during development
    )
