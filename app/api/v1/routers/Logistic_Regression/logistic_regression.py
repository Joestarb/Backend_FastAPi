from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict
from fastapi.responses import JSONResponse, FileResponse
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
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

    valid_features = {k: v for k, v in data.features.items() if k in df.columns}
    if not valid_features:
        raise HTTPException(status_code=400, detail="Ninguna columna ingresada está disponible en el CSV")

    X_cols = list(valid_features.keys())
    X = df[X_cols].copy()
    y = df[y_col].copy()

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

    try:
        input_vector = [valid_features[col] for col in X_cols]
        pred_class = int(model.predict([input_vector])[0])
        pred_prob = float(model.predict_proba([input_vector])[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {str(e)}")

    if pred_prob < 0.2:
        risk_level = 1
        description = "Riesgo extremadamente bajo según los factores evaluados."
    elif pred_prob < 0.4:
        risk_level = 3
        description = "Riesgo bajo, pero hay indicios a considerar."
    elif pred_prob < 0.6:
        risk_level = 5
        description = "Riesgo moderado, hay señales que requieren atención."
    elif pred_prob < 0.8:
        risk_level = 7
        description = "Riesgo elevado, se recomienda tomar medidas preventivas."
    else:
        risk_level = 9
        description = "Riesgo muy alto, se requiere intervención inmediata."

    analisis_columnas = []

    if 'Age' in valid_features:
        age = valid_features['Age']
        if age < 18:
            analisis_columnas.append("Edad menor a 18, población adolescente.")
        elif age <= 25:
            analisis_columnas.append("Joven adulto en etapa educativa.")
        else:
            analisis_columnas.append("Edad adulta, posiblemente educación superior o tardía.")

    if 'Avg_Daily_Usage_Hours' in valid_features:
        usage = valid_features['Avg_Daily_Usage_Hours']
        if usage <= 2:
            analisis_columnas.append("Uso de redes sociales bajo.")
        elif usage <= 5:
            analisis_columnas.append("Uso moderado de redes sociales.")
        else:
            analisis_columnas.append("Alto uso diario de redes sociales, posible dependencia.")

    if 'Mental_Health_Score' in valid_features:
        mental = valid_features['Mental_Health_Score']
        if mental >= 8:
            analisis_columnas.append("Salud mental excelente.")
        elif mental >= 5:
            analisis_columnas.append("Salud mental aceptable.")
        else:
            analisis_columnas.append("Indicadores de problemas de salud mental.")

    if 'Addicted_Score' in valid_features:
        addiction = valid_features['Addicted_Score']
        if addiction <= 3:
            analisis_columnas.append("Bajo nivel de adicción a redes.")
        elif addiction <= 7:
            analisis_columnas.append("Nivel de adicción moderado.")
        else:
            analisis_columnas.append("Nivel de adicción alto, indica posible dependencia.")

    if 'Conflicts_Over_Social_Media' in valid_features:
        conflict = valid_features['Conflicts_Over_Social_Media']
        if conflict == 0:
            analisis_columnas.append("Sin conflictos asociados al uso de redes.")
        elif conflict <= 5:
            analisis_columnas.append("Conflictos sociales moderados.")
        else:
            analisis_columnas.append("Frecuentes conflictos por redes sociales.")

    analisis_target = f"Se ha determinado que {'sí aplica' if pred_class == 1 else 'no aplica'}, con una probabilidad de {pred_prob:.2%}."

    os.makedirs(GRAPH_DIR, exist_ok=True)
    if os.path.exists(GRAPH_PATH):
        os.remove(GRAPH_PATH)

    probs = model.predict_proba([input_vector])[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.barplot(
        x=["No aplica", "Sí aplica"],
        y=probs,
        palette=["#f87171", "#60a5fa"],
        ax=ax
    )

    for i, p in enumerate(probs):
        ax.text(i, p + 0.02, f"{p:.2%}", ha='center', fontsize=13, fontweight='bold')

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Probabilidad estimada", fontsize=12)
    ax.set_title("Resultado del modelo de regresión logística", fontsize=14)
    ax.set_xlabel("Resultado", fontsize=12)
    ax.text(0.5, -0.15,
             f"Se ha determinado que {'sí aplica' if pred_class == 1 else 'no aplica'}, con una probabilidad de {pred_prob:.2%}.",
            ha='center', va='center', fontsize=10, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(GRAPH_PATH)
    plt.close()

    return JSONResponse({
        "features_utilizados": X_cols,
        "prediccion_clase": pred_class,
        "descripcion_prediccion": "Sí aplica" if pred_class == 1 else "No aplica",
        "probabilidad_clase_positiva": round(pred_prob, 4),
        "nivel_de_riesgo": risk_level,
        "interpretacion_general": description,
        "analisis_individual": analisis_columnas,
        "comentario_target": analisis_target,
        "grafica_url": "/predict/logistic-multiple/plot"
    })

@router.get("/predict/logistic-multiple/plot")
def get_logistic_multi_plot():
    if os.path.exists(GRAPH_PATH):
        return FileResponse(GRAPH_PATH, media_type="image/png")
    raise HTTPException(status_code=404, detail="No hay gráfica generada")
