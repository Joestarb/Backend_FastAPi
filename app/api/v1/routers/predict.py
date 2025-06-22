from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os

router = APIRouter()

CSV_DIR = os.path.join(os.getcwd(), "CSV")

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
