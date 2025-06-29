from fastapi import UploadFile, File, APIRouter, HTTPException
from fastapi.responses import JSONResponse
import shutil, os, csv
import pandas as pd

router = APIRouter()

CSV_FOLDER = os.path.join(os.getcwd(), "CSV")
PRUEBA_PATH = os.path.join(CSV_FOLDER, "Prueba.csv")
ENCODED_PATH = os.path.join(CSV_FOLDER, "datos_encoded.csv")

@router.post("/upload/csv")
async def upload_and_process_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .csv")

    os.makedirs(CSV_FOLDER, exist_ok=True)

    # Eliminar archivos previos
    for path in [PRUEBA_PATH, ENCODED_PATH]:
        if os.path.exists(path):
            os.remove(path)

    try:
        # Guardar archivo
        with open(PRUEBA_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df = pd.read_csv(PRUEBA_PATH)
        df.columns = df.columns.str.strip()

        required_cols = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Conflicts_Over_Social_Media']
        for col in required_cols:
            if col not in df.columns:
                raise HTTPException(status_code=422, detail=f"Falta la columna: {col}")

        # Nuevas columnas
        mental_scores = []
        addicted_scores = []
        academic_impacts = []

        for _, row in df.iterrows():
            usage = float(row['Avg_Daily_Usage_Hours'])
            sleep = float(row['Sleep_Hours_Per_Night'])
            conflicts = int(row['Conflicts_Over_Social_Media'])

            # --- Mental Health Score ---
            if sleep >= 8:
                mental = 9
            elif sleep >= 6:
                mental = 7
            else:
                mental = 4

            if usage > 6:
                mental -= 1
            if conflicts >= 3:
                mental -= 1
            mental = max(1, min(mental, 10))

            # --- Addicted Score ---
            if usage >= 7:
                addicted = 9
            elif usage >= 4:
                addicted = 6
            else:
                addicted = 3

            if conflicts > 3:
                addicted += 1
            if sleep > 7.5:
                addicted -= 1
            addicted = max(1, min(addicted, 10))

            # --- Affects Academic Performance ---
            impact = (
                (mental < 5 and addicted > 6) or
                (addicted >= 9) or
                (mental <= 4 and conflicts >= 3)
            )

            mental_scores.append(mental)
            addicted_scores.append(addicted)
            academic_impacts.append(impact)

        df['Mental_Health_Score'] = mental_scores
        df['Addicted_Score'] = addicted_scores
        df['Affects_Academic_Performance'] = academic_impacts

        df.to_csv(ENCODED_PATH, index=False)

        return {"message": "Archivo procesado exitosamente con valores fijos."}

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
