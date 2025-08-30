from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def root():
    return {"message": "Research Intelligence System up and running"}
