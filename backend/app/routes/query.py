from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def test_query():
    return {"message": "Query route is working!"}
