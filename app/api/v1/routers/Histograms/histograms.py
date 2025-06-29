

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Optional

router = APIRouter()

CSV_DIR = os.path.join(os.getcwd(), "CSV")
GRAPH_DIR = os.path.join(os.getcwd(), "Grafics", "Histograms")
os.makedirs(GRAPH_DIR, exist_ok=True)

def get_latest_csv_path():
    try:
        files = [f for f in os.listdir(CSV_DIR) if f.endswith('.csv')]
        if not files:
            raise FileNotFoundError("No hay archivos CSV en la carpeta CSV.")
        return os.path.join(CSV_DIR, files[0])
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

# --- ENDPOINT 1: USAGE HOURS HISTOGRAM + PREDICCIÓN ---
@router.get("/histogram/usage-hours")
def histogram_usage_hours(
    user_value: Optional[float] = Query(None, description="Valor del usuario para predecir su posición en el histograma"),
    bins: int = Query(14, description="Número de bins para el histograma"),
    color: str = Query('coral', description="Color de las barras"),
    edgecolor: str = Query('black', description="Color del borde de las barras")
):
    """
    Histograma de la distribución de horas de uso diario de redes sociales.
    Si se proporciona user_value, se marca en el histograma y se predice en qué percentil está.
    """
    try:
        csv_path = get_latest_csv_path()
        df = pd.read_csv(csv_path)
        data = df['Avg_Daily_Usage_Hours'].dropna()
        if data.empty:
            raise HTTPException(status_code=400, detail="No hay datos válidos en 'Avg_Daily_Usage_Hours'.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar datos: {str(e)}")

    plt.figure(figsize=(9, 5))
    plt.hist(data, bins=bins, color=color, edgecolor=edgecolor)
    if user_value is not None:
        plt.axvline(user_value, color='blue', linestyle='dashed', linewidth=2, label=f'Tu valor: {user_value}')
        plt.legend()
    plt.title('Distribución de Horas de Uso Diario')
    plt.xlabel('Horas por día')
    plt.ylabel('Cantidad de estudiantes')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    img_path = os.path.join(GRAPH_DIR, "hist_usage_hours.png")
    plt.savefig(img_path)
    plt.close()

    descripcion = (
        "Este histograma muestra cómo se distribuyen las horas de uso diario de redes sociales entre los estudiantes. "
        "Permite identificar si la mayoría pasa poco o mucho tiempo en redes sociales y detectar posibles extremos."
    )

    percentil = None
    interpretacion = None
    if user_value is not None:
        percentil = round((data < user_value).mean() * 100, 2)
        interpretacion = (
            f"Tu valor ({user_value} horas/día) está en el percentil {percentil} de la muestra. "
            f"Esto significa que aproximadamente el {percentil}% de los estudiantes usan menos horas que tú."
        )

    return JSONResponse({
        "grafica_url": "/api/v1/histogram/plot?hist_type=usage_hours",
        "descripcion": descripcion,
        "user_value": user_value,
        "percentil": percentil,
        "interpretacion": interpretacion
    })

# --- ENDPOINT 2: SLEEP ACADEMIC HISTOGRAM + PREDICCIÓN ---
@router.get("/histogram/sleep-academic")
def histogram_sleep_academic(
    user_value: Optional[float] = Query(None, description="Valor del usuario para predecir su posición en el histograma"),
    bins: int = Query(10, description="Número de bins para el histograma"),
    color_highschool: str = Query('red', description="Color para preparatoria"),
    color_undergrad: str = Query('yellow', description="Color para licenciatura"),
    color_grad: str = Query('blue', description="Color para posgrado"),
    academic_level: Optional[str] = Query(
        None,
        description="Nivel académico para la predicción. Opciones: Graduate, High School, Undergraduate"
    )
):
    """
    Comparación de horas de sueño por nivel académico. Si se da user_value y academic_level, se marca y predice el percentil.
    """
    try:
        csv_path = get_latest_csv_path()
        df = pd.read_csv(csv_path)
        # Usar columnas codificadas como booleanas
        hs = df[df['Academic_Level_High School'] == True]['Sleep_Hours_Per_Night'].dropna()
        ug = df[df['Academic_Level_Undergraduate'] == True]['Sleep_Hours_Per_Night'].dropna()
        grad = df[df['Academic_Level_Graduate'] == True]['Sleep_Hours_Per_Night'].dropna()
        if hs.empty and ug.empty and grad.empty:
            raise HTTPException(status_code=400, detail="No hay datos válidos para los niveles académicos.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al cargar datos: {str(e)}")

    plt.figure(figsize=(10,6))
    if not hs.empty:
        plt.hist(hs, bins=bins, alpha=0.6, label='Preparatoria', color=color_highschool, edgecolor='black')
    if not ug.empty:
        plt.hist(ug, bins=bins, alpha=0.6, label='Licenciatura', color=color_undergrad, edgecolor='black')
    if not grad.empty:
        plt.hist(grad, bins=bins, alpha=0.6, label='Posgrado', color=color_grad, edgecolor='black')
    if user_value is not None and academic_level:
        plt.axvline(user_value, color='blue', linestyle='dashed', linewidth=2, label=f'Tu valor: {user_value}')
    plt.title('Comparativa de Sueño por nivel académico')
    plt.xlabel('Horas dormidas')
    plt.ylabel('Cantidad de estudiantes')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    img_path = os.path.join(GRAPH_DIR, "hist_sleep_academic.png")
    plt.savefig(img_path)
    plt.close()

    descripcion = (
        "Este histograma compara las horas de sueño entre estudiantes de diferentes niveles académicos. "
        "Permite identificar si hay diferencias significativas en los hábitos de sueño según el nivel de estudios."
    )

    percentil = None
    interpretacion = None
    if user_value is not None and academic_level:
        academic_level_map = {
            'High School': hs,
            'Undergraduate': ug,
            'Graduate': grad
        }
        grupo = academic_level_map.get(academic_level)
        if grupo is None or grupo.empty:
            raise HTTPException(status_code=400, detail="El nivel académico debe ser uno de: Graduate, High School, Undergraduate.")
        percentil = round((grupo < user_value).mean() * 100, 2)
        interpretacion = (
            f"Tu valor ({user_value} horas de sueño) está en el percentil {percentil} de estudiantes de {academic_level}. "
            f"Esto significa que aproximadamente el {percentil}% de los estudiantes de ese nivel duermen menos que tú."
        )

    return JSONResponse({
        "grafica_url": "/api/v1/histogram/plot?hist_type=sleep_academic",
        "descripcion": descripcion,
        "user_value": user_value,
        "academic_level": academic_level,
        "percentil": percentil,
        "interpretacion": interpretacion
    })

@router.get("/histogram/plot")
def get_histogram_plot(hist_type: str = Query(..., description="Tipo de histograma: usage_hours, sleep_academic")):
    filename_map = {
        "usage_hours": "hist_usage_hours.png",
        "sleep_academic": "hist_sleep_academic.png"
    }
    img_file = filename_map.get(hist_type)
    if not img_file:
        raise HTTPException(status_code=400, detail="Tipo de histograma no válido.")
    img_path = os.path.join(GRAPH_DIR, img_file)
    if os.path.exists(img_path):
        return FileResponse(img_path, media_type='image/png')
    raise HTTPException(status_code=404, detail="No hay gráfica generada para este histograma.")
