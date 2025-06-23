from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

router = APIRouter()

CSV_DIR = os.path.join(os.getcwd(), "CSV")
GRAPH_DIR = os.path.join(os.getcwd(), "Grafics", "KMeans")
GRAPH_PATH = os.path.join(GRAPH_DIR, "kmeans_plot.png")

class UserInput(BaseModel):
    values: List[float]

class KMeansRequest(BaseModel):
    features: List[str]
    k: int
    normalize: bool = False
    user_input: Optional[UserInput] = None  # Nuevo campo para datos del usuario

def get_latest_csv_path():
    files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No hay archivos CSV.")
    return os.path.join(CSV_DIR, files[0])

def is_binary(series):
    unique_values = set(series.dropna().unique())
    return unique_values.issubset({0, 1})

@router.post("/predict/kmeans")
def kmeans_clustering(data: KMeansRequest):
    try:
        path = get_latest_csv_path()
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar CSV: {str(e)}")

    # Verificar columnas
    for col in data.features:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Columna '{col}' no encontrada en el CSV")

    # Verificar número de características
    if len(data.features) not in [2, 3]:
        raise HTTPException(status_code=400, detail="El número de características debe ser 2 o 3 para la visualización")

    # Preprocesamiento
    X_raw = df[data.features].dropna()
    
    # Identificar columnas binarias y no binarias
    binary_cols = [col for col in data.features if is_binary(X_raw[col])]
    non_binary_cols = [col for col in data.features if not is_binary(X_raw[col])]
    
    # Convertir a numéricos
    try:
        X = X_raw.copy()
        if non_binary_cols:
            X[non_binary_cols] = X[non_binary_cols].apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al convertir datos a numéricos: {str(e)}")

    # Verificar suficientes muestras
    if X.shape[0] < data.k:
        raise HTTPException(status_code=400, detail="No hay suficientes muestras para el número de clusters solicitado")

    # Normalización condicional
    if data.normalize and non_binary_cols:
        scaler = StandardScaler()
        X[non_binary_cols] = scaler.fit_transform(X[non_binary_cols])
        normalization_applied = True
    else:
        normalization_applied = False
        scaler = None

    # Clustering
    try:
        model = KMeans(n_clusters=data.k, random_state=42)
        clusters = model.fit_predict(X)
        X["cluster"] = clusters
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el clustering: {str(e)}")

    # Procesar datos del usuario si existen
    user_cluster = None
    if data.user_input:
        try:
            if len(data.user_input.values) != len(data.features):
                raise HTTPException(status_code=400, detail="El número de valores del usuario no coincide con las características seleccionadas")
            
            user_df = pd.DataFrame([data.user_input.values], columns=data.features)
            
            # Aplicar misma normalización si está activa
            if normalization_applied and non_binary_cols and scaler:
                user_df[non_binary_cols] = scaler.transform(user_df[non_binary_cols])
            
            # Predecir cluster del usuario
            user_cluster = int(model.predict(user_df)[0])
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al procesar datos del usuario: {str(e)}")

    # Visualización
    os.makedirs(GRAPH_DIR, exist_ok=True)
    if os.path.exists(GRAPH_PATH):
        os.remove(GRAPH_PATH)

    try:
        plt.figure(figsize=(10, 8))
        
        if len(data.features) == 2:
            # Gráfico 2D
            for cluster_id in range(data.k):
                subset = X[X["cluster"] == cluster_id]
                plt.scatter(subset[data.features[0]], subset[data.features[1]], 
                           label=f'Cluster {cluster_id}', alpha=0.7)
            
            # Marcar posición del usuario si existe
            if data.user_input:
                plt.scatter(user_df[data.features[0]], user_df[data.features[1]], 
                           c='red', s=200, marker='X', label='Tú')
            
            plt.xlabel(data.features[0])
            plt.ylabel(data.features[1])
            plt.title(f'K-Means Clustering ({data.k} grupos)')
            plt.legend()
            
        elif len(data.features) == 3:
            # Gráfico 3D
            ax = plt.axes(projection='3d')
            for cluster_id in range(data.k):
                subset = X[X["cluster"] == cluster_id]
                ax.scatter3D(subset[data.features[0]], 
                            subset[data.features[1]], 
                            subset[data.features[2]], 
                            label=f'Cluster {cluster_id}', alpha=0.7)
            
            # Marcar posición del usuario si existe
            if data.user_input:
                ax.scatter3D(user_df[data.features[0]], 
                            user_df[data.features[1]], 
                            user_df[data.features[2]], 
                            c='red', s=200, marker='X', label='Tú')
            
            ax.set_xlabel(data.features[0])
            ax.set_ylabel(data.features[1])
            ax.set_zlabel(data.features[2])
            ax.set_title(f'K-Means Clustering ({data.k} grupos)')
            ax.legend()
        
        plt.grid(True)
        plt.savefig(GRAPH_PATH)
        plt.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar gráfica: {str(e)}")

    # Preparar respuesta
    response = {
        "grafica_url": "/predict/kmeans/plot",
        "centroides": model.cluster_centers_.tolist(),
        "features_utilizados": data.features,
        "normalizacion_aplicada": normalization_applied,
        "columnas_binarias": binary_cols,
        "columnas_no_binarias": non_binary_cols
    }

    # Añadir información del usuario si existe
    if data.user_input:
        response["user_info"] = {
            "cluster_asignado": user_cluster,
            "valores_ingresados": data.user_input.values,
            "descripcion": f"Perteneces al Cluster {user_cluster}"
        }

    return JSONResponse(response)

@router.get("/predict/kmeans/plot")
def get_kmeans_plot():
    if os.path.exists(GRAPH_PATH):
        return FileResponse(GRAPH_PATH, media_type="image/png")
    raise HTTPException(status_code=404, detail="No hay gráfica generada")