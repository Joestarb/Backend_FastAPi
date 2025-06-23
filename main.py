from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Routers
from app.api.v1.routers import users
from app.api.v1.routers.Upload_CSV import upload_csv
from app.api.v1.routers.Logistic_Regression import logistic_regression
from app.api.v1.routers.Linear_Regression import predict
from app.api.v1.routers.KMeans_Clustering import kmeans_clustering  # NUEVO IMPORT

# Swagger declarado en raíz
app = FastAPI(
    title="Tu API",
    description="Documentación de la API",
    version="1.0",
    docs_url="/",  
    redoc_url=None  
)

# Configuración del CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*']
)

# =====================
# Endpoints de Usuarios
# =====================
app.include_router(users.router, prefix="/api/v1", tags=["Usuarios"])

# =====================
# Endpoints de Carga de CSV
# =====================
app.include_router(upload_csv.router, prefix="/api/v1", tags=["Carga de CSV"])

# =====================
# Endpoints de Regresión Lineal
# =====================
app.include_router(predict.router, prefix="/api/v1", tags=["Regresión Lineal"])

# =====================
# Endpoints de Regresión Logística
# =====================
app.include_router(logistic_regression.router, prefix="/api/v1", tags=["Regresión Logística"])

# ============================
# Endpoints de K-Means Clustering
# ============================
app.include_router(kmeans_clustering.router, prefix="/api/v1", tags=["K-Means Clustering"])  # NUEVO ENDPOINT
