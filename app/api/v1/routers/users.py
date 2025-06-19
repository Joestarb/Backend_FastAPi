from fastapi import APIRouter

router = APIRouter()

@router.get("/users")
def get_users():
    return ["user1", "user2"]
