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
        # Entrenamiento del modelo
        X = df[["Avg_Daily_Usage_Hours"]].values
        y = df["Mental_Health_Score"].values 
        model = LinearRegression()
        model.fit(X, y)

        # Predicción (ya en escala 0-10)
        pred = model.predict(np.array([[avg_daily_usage]]))[0]
        coef = model.coef_[0]
        intercept = model.intercept_

        # Graficar con más detalles
        plt.figure(figsize=(12, 8))
        
        # Datos y línea de regresión
        plt.scatter(X, y, color='blue', alpha=0.6, label='Datos reales')
        plt.plot(X, model.predict(X), color='red', linewidth=2, label=f'Regresión lineal (y = {coef:.2f}x + {intercept:.2f})')
        
        # Predicción
        plt.scatter([avg_daily_usage], [pred], color='green', s=200, marker='*', 
                   label=f'Predicción: {pred:.1f} pts\n({avg_daily_usage} hrs/día)')
        
        # Líneas de referencia (ajustadas a escala 0-10)
        plt.axhline(y=4, color='orange', linestyle='--', alpha=0.5, label='Límite salud baja')
        plt.axhline(y=7, color='purple', linestyle='--', alpha=0.5, label='Límite salud alta')
        
        # Zonas coloreadas (ajustadas a escala 0-10)
        plt.fill_between(X.flatten(), 0, 4, color='red', alpha=0.1, label='Salud mental baja')
        plt.fill_between(X.flatten(), 4, 7, color='yellow', alpha=0.1, label='Salud mental moderada')
        plt.fill_between(X.flatten(), 7, 10, color='green', alpha=0.1, label='Salud mental alta')
        
        # Configuración del gráfico con escala 0-10
        plt.xlabel('Horas promedio de uso diario de redes sociales', fontsize=12)
        plt.ylabel('Puntaje de salud mental (0-10)', fontsize=12)
        plt.title('Impacto del uso de redes sociales en la salud mental', fontsize=14, pad=20)
        plt.ylim(0, 10)  # Fijar límites del eje Y
        plt.yticks(np.arange(0, 11, 1))  # Marcas cada 1 punto
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Guardar gráfico
        img_path = 'prediction.png'
        plt.savefig(img_path, bbox_inches='tight', dpi=120)
        plt.close()

        # Interpretación detallada (usando escala 0-10)
        if pred < 4:
            risk_level = "Alto riesgo"
            recommendation = [
                "Reducir significativamente el uso diario de redes sociales",
                "Establecer horarios específicos para el uso de dispositivos",
                "Practicar actividades al aire libre y ejercicio regular",
                "Considerar apoyo profesional si persisten síntomas de ansiedad o depresión"
            ]
        elif 4 <= pred < 7:
            risk_level = "Riesgo moderado"
            recommendation = [
                "Mantener el uso por debajo de 4 horas diarias",
                "Implementar pausas activas cada 30 minutos de uso",
                "Practicar técnicas de mindfulness o meditación",
                "Fomentar interacciones sociales presenciales"
            ]
        else:
            risk_level = "Bajo riesgo"
            recommendation = [
                "Mantener buenos hábitos digitales",
                "Monitorear periódicamente el tiempo de uso",
                "Equilibrar actividades digitales con otras actividades recreativas",
                "Fomentar relaciones interpersonales fuera de línea"
            ]

        # Estadísticas adicionales (convertidas a escala 0-10)
        stats = {
            "correlacion": round(df['Avg_Daily_Usage_Hours'].corr(df['Mental_Health_Score']/10), 2),
            "uso_promedio_dataset": round(df['Avg_Daily_Usage_Hours'].mean(), 2),
            "puntaje_promedio_dataset": round(df['Mental_Health_Score'].mean()/10, 2),
            "coeficiente_regresion": round(coef, 4),
            "intercepto": round(intercept, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción o graficación: {str(e)}")

    return JSONResponse({
        "prediccion": round(float(pred), 2),
        "nivel_riesgo": risk_level,
        "rango_horas_ingresado": f"{avg_daily_usage} horas/día",
        "interpretacion": {
            "descripcion": "El puntaje de salud mental se mide en una escala de 0-10, donde valores más altos indican mejor salud mental.",
            "evaluacion": f"Con {avg_daily_usage} horas de uso diario, el modelo predice un puntaje de {round(pred, 1)}/10 ({risk_level}).",
            "recomendaciones": recommendation
        },
        "estadisticas": stats,
        "grafica_url": "/predict/mental_health/plot",
        "modelo_info": {
            "tipo": "Regresión Lineal",
            "ecuacion": f"y = {round(coef, 2)}x + {round(intercept, 2)}",
            "interpretacion_coeficiente": f"Cada hora adicional de uso diario se asocia con un cambio de {round(coef, 2)} puntos en el puntaje de salud mental (escala 0-10)."
        }
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

    # Generar resumen según el valor predicho
    if pred < 2:
        resumen = (
            f"El valor predicho de '{y_col}' es bajo ({round(pred, 2)}). "
            "Considera revisar los factores ingresados para mejorar este resultado."
        )
    elif 2 <= pred < 5:
        resumen = (
            f"El valor predicho de '{y_col}' es moderado ({round(pred, 2)}). "
            "Hay oportunidad de mejora ajustando los valores de entrada."
        )
    else:
        resumen = (
            f"El valor predicho de '{y_col}' es alto ({round(pred, 2)}). "
            "¡Excelente! Los factores seleccionados contribuyen positivamente."
        )

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
        "grafica_url": "/predict/linear-multiple/plot",
        "resumen": resumen
    })


@router.get("/predict/linear-multiple/plot")
def get_linear_multi_plot():
    img_path = os.path.join(GRAPH_DIR, "linear_plot_multi.png")
    if os.path.exists(img_path):
        return FileResponse(img_path, media_type="image/png")
    raise HTTPException(status_code=404, detail="No hay gráfica generada")

@router.get('/predict/linear-dynamic')
def predict_linear_dynamic(
    feature_col: str = Query(..., description="Nombre de la columna feature (X)"),
    target_col: str = Query(..., description="Nombre de la columna target (y)"),
    feature_value: float = Query(..., description="Valor de la feature para predecir")
):
    try:
        csv_path = get_latest_csv_path()
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar el CSV: {str(e)}")

    if feature_col not in df.columns or target_col not in df.columns:
        raise HTTPException(status_code=400, detail="Las columnas especificadas no existen en el CSV.")

    try:
        X = df[[feature_col]].values
        y = df[target_col].values
        model = LinearRegression()
        model.fit(X, y)
        pred = model.predict(np.array([[feature_value]]))[0]

        # Graficar
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y, color='blue', label='Datos reales')
        plt.plot(X, model.predict(X), color='red', label='Regresión lineal')
        plt.scatter([feature_value], [pred], color='green', label='Predicción')
        plt.xlabel(feature_col)
        plt.ylabel(target_col)
        plt.title(f'Predicción de {target_col} vs {feature_col}')
        plt.legend()
        img_path = 'prediction.png'
        plt.savefig(img_path)
        plt.close()

        resumen = f"El valor predicho de '{target_col}' para {feature_col}={feature_value} es {round(float(pred), 2)}."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante la predicción o graficación: {str(e)}")

    return JSONResponse({
        "prediccion": round(float(pred), 2),
        "grafica_url": "/predict/mental_health/plot",
        "resumen": resumen
    })
