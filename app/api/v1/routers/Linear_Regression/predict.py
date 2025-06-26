from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import os
from pydantic import BaseModel
from typing import Dict
import math
from  pydantic import BaseModel
from io import BytesIO
import base64
class PredictionRequest(BaseModel):
    x_values: list[float]
    y_values: list[float]
    x_to_predict: float

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



@router.post("/predict/linear-multiple")
def linear_multiple(data: LinearMultiFeatureRequest):
    """
    Endpoint para realizar predicciones utilizando un modelo de regresión lineal múltiple.

    Este endpoint permite entrenar un modelo de regresión lineal múltiple basado en un conjunto de datos cargado desde un archivo CSV.
    Luego, realiza una predicción utilizando las características proporcionadas por el usuario.

    Parámetros:
    - `data` (LinearMultiFeatureRequest): Objeto que contiene:
        - `features` (Dict[str, float]): Un diccionario donde las claves son los nombres de las columnas de características (X) y los valores son los valores correspondientes.
        - `target` (str): El nombre de la columna objetivo (y) en el conjunto de datos.

    Respuesta:
    - `features_utilizados` (list[str]): Lista de las características utilizadas para entrenar el modelo.
    - `prediccion` (float): El valor predicho para la columna objetivo basado en las características proporcionadas.
    - `grafica_url` (str): URL para acceder a la gráfica generada.
    - `resumen` (str): Un resumen interpretativo del valor predicho.

    Errores:
    - 400: Si la columna objetivo no es válida o si ninguna de las características proporcionadas está disponible en el CSV.
    - 500: Si ocurre un error al cargar el CSV, entrenar el modelo o realizar la predicción.

    Notas:
    - El modelo se entrena dinámicamente cada vez que se llama al endpoint.
    - Si hay más de dos características, no se genera una gráfica.
    """
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

@router.get('/predict/linear-multiple/plot')
def get_linear_multiple_plot():
    img_path = os.path.join(GRAPH_DIR, "linear_plot_multi.png")
    if os.path.exists(img_path):
        return FileResponse(img_path, media_type='image/png')
    raise HTTPException(status_code=404, detail="No hay gráfica generada")
  


def sanitize_json(data):
    if isinstance(data, dict):
        return {k: sanitize_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_json(v) for v in data]
    elif isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
        return None  # Replace invalid float values with None
    return data

@router.get('/social-media/analysis')
def get_social_media_analysis(
    age_range: str = Query("18-25", description="Rango de edad (formato 'min-max')"),
    country: str = Query(None, description="País a filtrar (opcional)"),
    academic_level: str = Query(None, description="Nivel académico", enum=["High School", "Undergraduate", "Graduate"])
):
    """
    Análisis del impacto de redes sociales basado en el dataset proporcionado.
    Proporciona estadísticas sobre:
    - Uso diario promedio
    - Efecto en rendimiento académico
    - Salud mental
    - Adicción
    - Plataformas más utilizadas
    Los datos incluyen correlaciones y análisis segmentados por:
    - Edad
    - País
    - Nivel académico
    """
    try:
        # Leer el archivo CSV desde la carpeta CSV
        csv_path = os.path.join(CSV_DIR, 'datos_encoded.csv')
        df = pd.read_csv(csv_path)

        # Procesar parámetros
        age_min, age_max = map(int, age_range.split('-'))
        filtered_df = df[(df['Age'] >= age_min) & (df['Age'] <= age_max)]

        if country:
            filtered_df = filtered_df[filtered_df['Country'] == country]

        if academic_level:
            level_col = f"Academic_Level_{academic_level.replace(' ', '_')}"
            if level_col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[level_col] == 1]

        # Cálculos estadísticos
        total_students = len(filtered_df)
        avg_usage = filtered_df['Avg_Daily_Usage_Hours'].mean()
        affects_performance = filtered_df['Affects_Academic_Performance'].value_counts(normalize=True).get(1, 0) * 100
        avg_mental_health = filtered_df['Mental_Health_Score'].mean()
        avg_addiction = filtered_df['Addicted_Score'].mean()
        avg_sleep = filtered_df['Sleep_Hours_Per_Night'].mean()

        # Plataformas más populares
        platform_cols = [col for col in df.columns if col.startswith('Most_Used_Platform_')]
        platform_usage = filtered_df[platform_cols].sum().sort_values(ascending=False)
        top_platforms = platform_usage.head(3).to_dict()

        # Correlaciones
        correlation_matrix = filtered_df[['Avg_Daily_Usage_Hours', 'Mental_Health_Score', 
                                         'Addicted_Score', 'Sleep_Hours_Per_Night']].corr()

        # Segmentación por horas de uso
        usage_bins = [0, 2, 4, 6, 8, 24]
        usage_labels = ['0-2h', '2-4h', '4-6h', '6-8h', '8+h']
        filtered_df['Usage_Category'] = pd.cut(filtered_df['Avg_Daily_Usage_Hours'], 
                                             bins=usage_bins, labels=usage_labels)

        usage_stats = filtered_df.groupby('Usage_Category').agg({
            'Mental_Health_Score': 'mean',
            'Addicted_Score': 'mean',
            'Affects_Academic_Performance': lambda x: (x == 1).mean() * 100
        }).reset_index().to_dict('records')

        # Generar gráfico de correlación
        import seaborn as sns
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        chart_path = os.path.join(GRAPH_DIR, 'social_media_correlation.png')
        plt.savefig(chart_path)
        plt.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando datos: {str(e)}")

    return JSONResponse(sanitize_json({
        "summary": {
            "total_students": total_students,
            "avg_daily_usage_hours": round(avg_usage, 1),
            "affects_performance_percentage": round(affects_performance, 1),
            "avg_mental_health_score": round(avg_mental_health, 1),
            "avg_addiction_score": round(avg_addiction, 1),
            "avg_sleep_hours": round(avg_sleep, 1)
        },
        "platform_analysis": {
            "top_platforms": top_platforms,
            "platform_usage_distribution": platform_usage.to_dict()
        },
        "usage_impact": {
            "by_usage_category": usage_stats,
            "correlation_usage_mental_health": round(correlation_matrix.loc['Avg_Daily_Usage_Hours', 'Mental_Health_Score'], 2),
            "correlation_usage_addiction": round(correlation_matrix.loc['Avg_Daily_Usage_Hours', 'Addicted_Score'], 2),
            "correlation_usage_sleep": round(correlation_matrix.loc['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night'], 2)
        },
        "demographic_filters": {
            "age_range": f"{age_min}-{age_max}",
            "country": country or "Todos",
            "academic_level": academic_level or "Todos"
        },
        "correlation_chart_url": "/social-media/analysis/correlation-chart",
        "insights": {
            "performance_impact": "El uso >4h/día muestra mayor impacto en rendimiento académico",
            "mental_health_tip": "Cada hora adicional de uso correlaciona con disminución en salud mental",
            "sleep_quality": "Menos horas de sueño correlacionan con mayor uso de redes sociales"
        }
    }))

@router.get('/social-media/analysis/correlation-chart')
def get_correlation_chart():
    chart_path = os.path.join(GRAPH_DIR, 'social_media_correlation.png')
    if os.path.exists(chart_path):
        return FileResponse(chart_path, media_type='image/png')
    raise HTTPException(status_code=404, detail="No hay gráfico generado")