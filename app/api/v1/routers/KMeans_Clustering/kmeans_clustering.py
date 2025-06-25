from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
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
    user_input: Optional[UserInput] = None

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
    if data.k > 5:
        raise HTTPException(status_code=400, detail="Solo se permiten hasta 5 clusters para mantener la interpretabilidad")

    try:
        path = get_latest_csv_path()
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar CSV: {str(e)}")

    for col in data.features:
        if col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Columna '{col}' no encontrada en el CSV")

    if len(data.features) not in [2, 3]:
        raise HTTPException(status_code=400, detail="El número de características debe ser 2 o 3 para la visualización")

    X_raw = df[data.features].dropna()
    binary_cols = [col for col in data.features if is_binary(X_raw[col])]
    non_binary_cols = [col for col in data.features if not is_binary(X_raw[col])]

    try:
        X = X_raw.copy()
        if non_binary_cols:
            X[non_binary_cols] = X[non_binary_cols].apply(pd.to_numeric, errors='coerce')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al convertir datos a numéricos: {str(e)}")

    if X.shape[0] < data.k:
        raise HTTPException(status_code=400, detail="No hay suficientes muestras para el número de clusters solicitado")

    if data.normalize and non_binary_cols:
        scaler = StandardScaler()
        X[non_binary_cols] = scaler.fit_transform(X[non_binary_cols])
        normalization_applied = True
    else:
        normalization_applied = False
        scaler = None

    try:
        model = KMeans(n_clusters=data.k, random_state=42)
        clusters = model.fit_predict(X)
        X["cluster"] = clusters
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el clustering: {str(e)}")

    user_cluster = None
    user_df = None
    if data.user_input:
        try:
            if len(data.user_input.values) != len(data.features):
                raise HTTPException(status_code=400, detail="Número de valores del usuario no coincide con las características")
            user_df = pd.DataFrame([data.user_input.values], columns=data.features)
            if normalization_applied and non_binary_cols and scaler:
                user_df[non_binary_cols] = scaler.transform(user_df[non_binary_cols])
            user_cluster = int(model.predict(user_df)[0])
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error al procesar datos del usuario: {str(e)}")

    etiquetas_es = {
        "Age": "Edad",
        "Avg_Daily_Usage_Hours": "Uso Diario Promedio",
        "Mental_Health_Score": "Índice de Salud Mental",
        "Addicted_Score": "Índice de Adicción",
        "Conflicts_Over_Social_Media": "Conflictos por Redes Sociales",
        "Sleep_Hours_Per_Night": "Horas de Sueño por Noche"
    }

    nombre_x = etiquetas_es.get(data.features[0], data.features[0])
    nombre_y = etiquetas_es.get(data.features[1], data.features[1])
    nombre_z = etiquetas_es.get(data.features[2], data.features[2]) if len(data.features) == 3 else None

    etiquetas_grupos = {
        0: "Grupo 0: Bajo uso y bajo riesgo",
        1: "Grupo 1: Uso moderado y riesgo medio",
        2: "Grupo 2: Alto uso y posible dependencia",
        3: "Grupo 3: Uso intenso y riesgo elevado",
        4: "Grupo 4: Uso extremo y problemas frecuentes"
    }
    etiquetas_usadas = {k: v for k, v in etiquetas_grupos.items() if k < data.k}

    os.makedirs(GRAPH_DIR, exist_ok=True)
    if os.path.exists(GRAPH_PATH):
        os.remove(GRAPH_PATH)

    try:
        plt.figure(figsize=(10, 8))

        if len(data.features) == 2:
            for cluster_id in range(data.k):
                subset = X[X["cluster"] == cluster_id]
                plt.scatter(
                    subset[data.features[0]], subset[data.features[1]],
                    label=etiquetas_usadas.get(cluster_id, f'Grupo {cluster_id}'), alpha=0.6
                )
            if data.user_input:
                plt.scatter(
                    user_df[data.features[0]], user_df[data.features[1]],
                    c='red', s=200, marker='X', label='Tú'
                )
                plt.text(
                    user_df[data.features[0]].values[0],
                    user_df[data.features[1]].values[0],
                    f'Pertenece a:\n{etiquetas_usadas.get(user_cluster, f"Grupo {user_cluster}")}',
                    fontsize=10, color='red', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red')
                )

            plt.xlabel(nombre_x)
            plt.ylabel(nombre_y)
            plt.title(f'Agrupamiento K-Means en {data.k} Grupos')
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

        elif len(data.features) == 3:
            ax = plt.axes(projection='3d')
            for cluster_id in range(data.k):
                subset = X[X["cluster"] == cluster_id]
                ax.scatter3D(
                    subset[data.features[0]], subset[data.features[1]], subset[data.features[2]],
                    label=etiquetas_usadas.get(cluster_id, f'Grupo {cluster_id}'), alpha=0.6
                )
            if data.user_input:
                ax.scatter3D(
                    user_df[data.features[0]], user_df[data.features[1]], user_df[data.features[2]],
                    c='red', s=200, marker='X', label='Tú'
                )
                ax.text2D(
                    0.05, 0.95,
                    f'Perteneces a: {etiquetas_usadas.get(user_cluster, f"Grupo {user_cluster}")}',
                    transform=ax.transAxes,
                    fontsize=10, color='red', weight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red')
                )

            ax.set_xlabel(nombre_x)
            ax.set_ylabel(nombre_y)
            ax.set_zlabel(nombre_z)
            ax.set_title(f'Agrupamiento K-Means en {data.k} Grupos')
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

        plt.grid(True)
        plt.tight_layout()
        plt.savefig(GRAPH_PATH)
        plt.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar gráfica: {str(e)}")

    analisis_columnas = []
    for i, col in enumerate(data.features):
        valor = data.user_input.values[i] if data.user_input else X[col].mean()
        if col == "Age":
            if valor < 18:
                analisis_columnas.append("Edad adolescente, posibles hábitos digitales en desarrollo.")
            elif valor <= 25:
                analisis_columnas.append("Joven adulto, comportamiento exploratorio común.")
            else:
                analisis_columnas.append("Edad adulta, hábitos más consolidados.")
        elif col == "Avg_Daily_Usage_Hours":
            if valor <= 2:
                analisis_columnas.append("Uso bajo de redes sociales.")
            elif valor <= 5:
                analisis_columnas.append("Uso moderado de redes sociales.")
            else:
                analisis_columnas.append("Uso intensivo de redes sociales, posible riesgo de dependencia.")
        elif col == "Mental_Health_Score":
            if valor >= 8:
                analisis_columnas.append("Buena salud mental.")
            elif valor >= 5:
                analisis_columnas.append("Salud mental aceptable, aunque con riesgos.")
            else:
                analisis_columnas.append("Posibles indicadores de riesgo psicológico.")
        elif col == "Addicted_Score":
            if valor <= 3:
                analisis_columnas.append("Bajo nivel de adicción a redes sociales.")
            elif valor <= 7:
                analisis_columnas.append("Nivel medio de adicción.")
            else:
                analisis_columnas.append("Nivel alto de adicción, atención recomendada.")
        elif col == "Conflicts_Over_Social_Media":
            if valor == 0:
                analisis_columnas.append("Sin conflictos asociados a redes.")
            elif valor <= 5:
                analisis_columnas.append("Algunos conflictos sociales moderados.")
            else:
                analisis_columnas.append("Conflictos frecuentes, impacto social detectado.")
        elif col == "Sleep_Hours_Per_Night":
            if valor < 5:
                analisis_columnas.append("Horas de sueño insuficientes, posible impacto en salud.")
            elif valor <= 7:
                analisis_columnas.append("Horas de sueño adecuadas para recuperación.")
            else:
                analisis_columnas.append("Horas de sueño por encima del promedio, podría indicar exceso.")

    response = {
        "grafica_url": "/predict/kmeans/plot",
        "features_utilizados": data.features,
        "normalizacion_aplicada": normalization_applied,
        "columnas_binarias": binary_cols,
        "columnas_no_binarias": non_binary_cols,
        "descripcion_cluster": analisis_columnas
    }

    if data.user_input:
        response["user_info"] = {
            "cluster_asignado": user_cluster,
            "valores_ingresados": data.user_input.values,
            "descripcion": f"Perteneces a: {etiquetas_usadas.get(user_cluster, f'Grupo {user_cluster}')}"
        }

    return JSONResponse(response)

@router.get("/predict/kmeans/plot")
def get_kmeans_plot():
    if os.path.exists(GRAPH_PATH):
        return FileResponse(GRAPH_PATH, media_type="image/png")
    raise HTTPException(status_code=404, detail="No hay gráfica generada")