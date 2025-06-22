from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import os

router = APIRouter()

CSV_DIR = os.path.join(os.getcwd(), "CSV")
GRAPH_DIR = os.path.join(os.getcwd(), "Grafics", "LogisticR")
GRAPH_PATH = os.path.join(GRAPH_DIR, "logistic_plot_multi.png")


class MultiFeatureRequest(BaseModel):
    features: Dict[str, float]
    target: str


def get_latest_csv_path():
    files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError("No hay archivos CSV.")
    return os.path.join(CSV_DIR, files[0])


@router.post("/predict/logistic-multiple")
def logistic_multiple(data: MultiFeatureRequest):
    try:
        path = get_latest_csv_path()
        df = pd.read_csv(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar CSV: {str(e)}")

    y_col = data.target

    if y_col not in df.columns:
        raise HTTPException(status_code=400, detail="Columna objetivo no válida")

    # Validar columnas que SÍ están en el CSV
    valid_features = {k: v for k, v in data.features.items() if k in df.columns}
    
    if not valid_features:
        raise HTTPException(status_code=400, detail="Ninguna columna ingresada está disponible en el CSV")

    X_cols = list(valid_features.keys())
    X = df[X_cols].copy()
    y = df[y_col].copy()

    # Convertir target a binario si es necesario
    if y.dtype == 'object':
        y = y.astype(str).str.lower().map({'yes': 1, 'no': 0, 'si': 1, 'sí': 1, 'true': 1, 'false': 0, '1': 1, '0': 0})
    elif y.dtype == 'bool':
        y = y.astype(int)

    if y.isnull().any() or not set(y.unique()).issubset({0, 1}):
        raise HTTPException(status_code=400, detail="La columna objetivo debe ser binaria")

    try:
        model = LogisticRegression()
        model.fit(X, y)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al entrenar el modelo: {str(e)}")

    # Vector de predicción con los mismos features
    try:
        input_vector = [valid_features[col] for col in X_cols]
        pred_class = int(model.predict([input_vector])[0])
        pred_prob = float(model.predict_proba([input_vector])[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {str(e)}")

    # Guardar la gráfica
    os.makedirs(GRAPH_DIR, exist_ok=True)
    if os.path.exists(GRAPH_PATH):
        os.remove(GRAPH_PATH)

    probs = model.predict_proba([input_vector])[0]
    classes = ['(No Aplica)', '(Sí Aplica)']
    colors = ['#6baed6', '#74c476']

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(classes, probs, color=colors, width=0.5)

    # Añadir etiquetas sobre las barras
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{prob:.2%}', ha='center', va='bottom', fontsize=12)

    ax.set_ylim(0, 1)
    ax.set_ylabel('Probabilidad', fontsize=12)
    ax.set_title(f"Probabilidades estimadas para '{y_col}'", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(GRAPH_PATH)
    plt.close()


    return JSONResponse({
        "features_utilizados": X_cols,
        "prediccion_clase": pred_class,
        "probabilidad_clase_positiva": round(pred_prob, 4),
        "grafica_url": "/predict/logistic-multiple/plot"
    })


@router.get("/predict/logistic-multiple/plot")
def get_logistic_multi_plot():
    if os.path.exists(GRAPH_PATH):
        return FileResponse(GRAPH_PATH, media_type="image/png")
    raise HTTPException(status_code=404, detail="No hay gráfica generada")
