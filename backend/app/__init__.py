# backend/app/__init__.py

from fastapi import FastAPI
from .routes import query, feedback

app = FastAPI(
    title="BookBuddy API",
    description="API for BookBuddy: AI-powered reading assistant",
    version="1.0.0"
)

# Register your routes (routers)
app.include_router(query.router, prefix="/query", tags=["Query"])
app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])
