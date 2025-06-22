from fastapi import FastAPI
from app.api.v1.routers import users, predict
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.routers.Upload_CSV import upload_csv
from app.api.v1.routers.Logistic_Regression import logistic_regression

# Swagger declarado en raiz
app = FastAPI(
    title="Tu API",
    description="Documentación de la API",
    version="1.0",
    docs_url="/",  
    redoc_url=None  
)

# Configuracion del cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)

# Declaracion del archivo que contiene rutas que debe reconocer el router
app.include_router(users.router, prefix="/api/v1")
app.include_router(predict.router, prefix="/api/v1")
app.include_router(upload_csv.router, prefix="/api/v1")
app.include_router(logistic_regression.router, prefix="/api/v1")
