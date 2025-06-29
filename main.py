from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Routers
from app.api.v1.routers import users
from app.api.v1.routers.Upload_CSV import upload_csv
from app.api.v1.routers.Logistic_Regression import logistic_regression
from app.api.v1.routers.Linear_Regression import predict
from app.api.v1.routers.KMeans_Clustering import kmeans_clustering  # NUEVO IMPORT
from app.api.v1.routers.BoxPlot import boxplot  # Importar el router de BoxPlot
from app.api.v1.routers.Prediction import prediction_user_form
from app.api.v1.routers.Histograms import histograms  # Importar el router de Histogramas

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
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
app.include_router(kmeans_clustering.router, prefix="/api/v1", tags=["K-Means Clustering"])
# ============================
# Endpoints de BoxPlot
# ============================
app.include_router(boxplot.router, prefix="/api/v1", tags=["BoxPlot"])
# =====================
# Endpoints de Predicción de Salud Mental y Adicción
# =====================
app.include_router(prediction_user_form.router, prefix="/api/v1", tags=["Predicción de Salud Mental Y adicción"])
# ============================
# Endpoints de Histogramas
# ============================
app.include_router(histograms.router, prefix="/api/v1", tags=["Histogramas"])