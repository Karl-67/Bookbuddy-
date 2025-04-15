from fastapi import APIRouter

router = APIRouter()

@router.post("/")
def submit_feedback(feedback: str):
    return {"message": "Feedback received!", "feedback": feedback}
