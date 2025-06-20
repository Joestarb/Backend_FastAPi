from fastapi import FastAPI
from app.api.v1.routers import users, predict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Tu API",
    description="Documentación de la API",
    version="1.0",
    docs_url="/",  # Swagger en la raíz
    redoc_url=None  # Deshabilita Redoc
    
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'])



app.include_router(users.router, prefix="/api/v1")
app.include_router(predict.router, prefix="/api/v1")