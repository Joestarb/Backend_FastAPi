from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from pydantic import BaseModel
from typing import Dict

router = APIRouter()

CSV_DIR = os.path.join(os.getcwd(), "CSV")
GRAPH_DIR = os.path.join(os.getcwd(), "Grafics", "LinearR")

class LinearMultiFeatureRequest(BaseModel):
    features: Dict[str, float]
    target: str


def get_latest_csv_path():
    try:
        files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
        if not files:
            raise FileNotFoundError("No hay archivos CSV en la carpeta CSV.")
        return os.path.join(CSV_DIR, files[0])
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get('/predict/mental_health')
def predict_mental_health(avg_daily_usage: float = Query(..., description="Horas promedio de uso diario de redes sociales")):
    try:
        csv_path = get_latest_csv_path()
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el CSV: {str(e)}")

    try:
        X = df[["Avg_Daily_Usage_Hours"]].values
        y = df["Mental_Health_Score"].values
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(np.array([[avg_daily_usage]]))[0]
        plt.figure(figsize=(8,5))
        plt.scatter(X, y, color='blue', label='Datos reales')
        plt.plot(X, model.predict(X), color='red', label='Regresión lineal')
        plt.scatter([avg_daily_usage], [pred], color='green', label='Predicción')
        plt.xlabel('Horas promedio de uso diario')
        plt.ylabel('Puntaje de salud mental')
        plt.title('Predicción de salud mental vs uso de redes sociales')
        plt.legend()
        img_path = 'prediction.png'
        plt.savefig(img_path)
        plt.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción o graficación: {str(e)}")

    return JSONResponse({
        "prediccion": pred,
        "grafica_url": "/predict/mental_health/plot"
    })


@router.get('/predict/mental_health/plot')
def get_plot():
    img_path = 'prediction.png'
    if os.path.exists(img_path):
        return FileResponse(img_path, media_type='image/png')
    raise HTTPException(status_code=404, detail="No hay gráfica generada")


@router.get('/predict/csv-heads')
def get_csv_heads():
    try:
        csv_path = get_latest_csv_path()
        df = pd.read_csv(csv_path)
        df = df.convert_dtypes()
        cols = df.columns.tolist()
        data_types = df.dtypes.astype(str).tolist()
        column_info = [{"columna": c, "tipo": t} for c, t in zip(cols, data_types)]
        return JSONResponse({"columnas": column_info})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo CSV: {str(e)}")


@router.get('/predict/modular-prediction')
def predict_modular_prediction(
    avg_daily_usage: float = Query(..., description="Horas promedio de uso diario de redes sociales"),
    heads: list[str] = Query(..., description="Lista de dos nombres de columnas del CSV en el orden [X, y]")
):
    if len(heads) != 2:
        raise HTTPException(status_code=400, detail="Debe proporcionar exactamente dos nombres de columnas.")

    try:
        csv_path = get_latest_csv_path()
        df = pd.read_csv(csv_path)
        X = df[[heads[0]]].values
        y = df[heads[1]].values

        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(np.array([[avg_daily_usage]]))[0]

        plt.figure(figsize=(8,5))
        plt.scatter(X, y, color='blue', label='Datos reales')
        plt.plot(X, model.predict(X), color='red', label='Regresión lineal')
        plt.scatter([avg_daily_usage], [pred], color='green', label='Predicción')
        plt.xlabel(heads[0])
        plt.ylabel(heads[1])
        plt.title(f'Predicción de {heads[1]} vs {heads[0]}')
        plt.legend()
        img_path = 'prediction.png'
        plt.savefig(img_path)
        plt.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante predicción o graficación: {str(e)}")

    return JSONResponse({
        "prediccion": pred,
        "grafica_url": "/predict/mental_health/plot"
    })


@router.post("/predict/linear-multiple")
def linear_multiple(data: LinearMultiFeatureRequest):
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

    try:
        model = LinearRegression()
        model.fit(X, y)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al entrenar el modelo: {str(e)}")

    # Vector de predicción con los mismos features
    try:
        input_vector = [valid_features[col] for col in X_cols]
        pred = float(model.predict([input_vector])[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción: {str(e)}")

    # Guardar la gráfica
    os.makedirs(GRAPH_DIR, exist_ok=True)
    img_path = os.path.join(GRAPH_DIR, "linear_plot_multi.png")
    if os.path.exists(img_path):
        os.remove(img_path)

    # Graficar solo si hay 1 o 2 features
    plt.figure(figsize=(8, 6))
    if len(X_cols) == 1:
        plt.scatter(X[X_cols[0]], y, color='blue', label='Datos reales')
        plt.plot(X[X_cols[0]], model.predict(X), color='red', label='Regresión lineal')
        plt.scatter([valid_features[X_cols[0]]], [pred], color='green', label='Predicción')
        plt.xlabel(X_cols[0])
        plt.ylabel(y_col)
    elif len(X_cols) == 2:
        from mpl_toolkits.mplot3d import Axes3D
        ax = plt.axes(projection='3d')
        ax.scatter(X[X_cols[0]], X[X_cols[1]], y, color='blue', label='Datos reales')
        ax.set_xlabel(X_cols[0])
        ax.set_ylabel(X_cols[1])
        ax.set_zlabel(y_col)
    else:
        plt.text(0.5, 0.5, 'No se puede graficar más de 2 features', ha='center', va='center', fontsize=14)
    plt.title(f"Predicción de {y_col} usando {', '.join(X_cols)}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(img_path)
    plt.close()

    return JSONResponse({
        "features_utilizados": X_cols,
        "prediccion": round(pred, 4),
        "grafica_url": "/predict/linear-multiple/plot"
    })


@router.get("/predict/linear-multiple/plot")
def get_linear_multi_plot():
    img_path = os.path.join(GRAPH_DIR, "linear_plot_multi.png")
    if os.path.exists(img_path):
        return FileResponse(img_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="No hay gráfica generada")
