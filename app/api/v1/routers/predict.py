from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os

router = APIRouter()
#ejemplo de como usar el endpoint: http://localhost:8000/api/v1/predict/mental_health?avg_daily_usage=5
# Ajustar la ruta para que apunte a la raíz del proyecto
CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../Social Media.csv'))

@router.get('/predict/mental_health')
def predict_mental_health(avg_daily_usage: float = Query(..., description="Horas promedio de uso diario de redes sociales")):
    # Leer el CSV
    df = pd.read_csv(CSV_PATH)
    X = df[["Avg_Daily_Usage_Hours"]].values
    y = df["Mental_Health_Score"].values

    # Entrenar modelo de regresión lineal
    model = LinearRegression()
    model.fit(X, y)

    # Realizar predicción
    pred = model.predict(np.array([[avg_daily_usage]]))[0]

    # Graficar
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

    # Responder con la predicción y la imagen
    return JSONResponse({
        "prediccion": pred,
        "grafica_url": "/predict/mental_health/plot"
    })

@router.get('/predict/mental_health/plot')
def get_plot():
    img_path = 'prediction.png'
    if os.path.exists(img_path):
        return FileResponse(img_path, media_type='image/png')
    return JSONResponse({"error": "No hay gráfica generada"}, status_code=404)
