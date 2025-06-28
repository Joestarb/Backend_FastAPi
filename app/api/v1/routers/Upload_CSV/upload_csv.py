from fastapi import UploadFile, File, APIRouter, HTTPException
from fastapi.responses import JSONResponse
import shutil, os, csv
import pandas as pd
import numpy as np
from pathlib import Path

router = APIRouter()

CSV_FOLDER = os.path.join(os.getcwd(), "CSV")
PRUEBA_PATH = os.path.join(CSV_FOLDER, "Prueba.csv")
ENCODED_PATH = os.path.join(CSV_FOLDER, "datos_encoded.csv")

@router.post("/upload/csv")
async def upload_and_process_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .csv")

    os.makedirs(CSV_FOLDER, exist_ok=True)

    # Limpiar carpeta
    for f in os.listdir(CSV_FOLDER):
        file_path = os.path.join(CSV_FOLDER, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    try:
        # Guardar como Prueba.csv
        with open(PRUEBA_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Leer CSV
        df = pd.read_csv(PRUEBA_PATH)

        # Listas para los nuevos campos
        mental_scores = []
        addicted_scores = []
        academic_impacts = []

        for _, row in df.iterrows():
            sleep = row['Sleep_Hours_Per_Night']
            usage = row['Avg_Daily_Usage_Hours']

            # Mental Health Score (ideal 8h)
            mental_score = 10 - abs(8 - sleep) * 1.5
            mental_score = max(1, min(mental_score, 10))

            # Addicted Score (máximo 10h)
            addicted_score = (min(usage, 10) / 10) * 10
            addicted_score = max(1, min(addicted_score, 10))

            # Impacto académico
            performance_impact = addicted_score > 7 and mental_score < 5

            mental_scores.append(round(mental_score, 2))
            addicted_scores.append(round(addicted_score, 2))
            academic_impacts.append(performance_impact)

        # Agregar columnas
        df['Mental_Health_Score'] = mental_scores
        df['Addicted_Score'] = addicted_scores
        df['Affects_Academic_Performance'] = academic_impacts

        # Guardar como datos_encoded.csv
        df.to_csv(ENCODED_PATH, index=False)

        return {"message": "Archivo guardado como 'Prueba.csv' y procesado como 'datos_encoded.csv'"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el archivo CSV: {str(e)}")


@router.get("/read/csv")
async def read_csv():
    try:
        if not os.path.exists(ENCODED_PATH):
            raise HTTPException(status_code=404, detail="No existe datos_encoded.csv")

        with open(ENCODED_PATH, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            data = [row for row in reader]

        return JSONResponse(content=data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer datos_encoded.csv: {str(e)}")
