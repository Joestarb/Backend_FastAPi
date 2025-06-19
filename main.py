from fastapi import FastAPI
from app.api.v1.routers import users, predict

app = FastAPI()

app.include_router(users.router, prefix="/api/v1")
app.include_router(predict.router, prefix="/api/v1")