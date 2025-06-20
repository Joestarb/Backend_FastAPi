from fastapi import FastAPI
from app.api.v1.routers import users, predict

app = FastAPI(
    title="Tu API",
    description="Documentación de la API",
    version="1.0",
    docs_url="/",  # Swagger en la raíz
    redoc_url=None  # Deshabilita Redoc
)

app.include_router(users.router, prefix="/api/v1")
app.include_router(predict.router, prefix="/api/v1")