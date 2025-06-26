from fastapi import UploadFile, File, APIRouter, HTTPException
import shutil, os
import csv
from fastapi.responses import JSONResponse

router = APIRouter()

CSV_FOLDER = os.path.join(os.getcwd(), "CSV")  # Ruta a la carpeta CSV en la raíz del proyecto

@router.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .csv")

    # Crear la carpeta CSV si no existe
    os.makedirs(CSV_FOLDER, exist_ok=True)

    # Eliminar archivos existentes en la carpeta CSV
    for f in os.listdir(CSV_FOLDER):
        file_path = os.path.join(CSV_FOLDER, f)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Ruta donde se guardará el nuevo archivo
    save_path = os.path.join(CSV_FOLDER, file.filename)

    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al guardar el archivo: {str(e)}")

    return {"message": f"Archivo '{file.filename}' guardado exitosamente en /CSV"}

@router.get("/read/csv")
async def read_csv():
    try:
        # Buscar el primer archivo CSV en la carpeta
        files = [f for f in os.listdir(CSV_FOLDER) if f.endswith('.csv')]
        if not files:
            raise HTTPException(status_code=404, detail="No hay archivos CSV en la carpeta")

        csv_path = os.path.join(CSV_FOLDER, files[0])

        # Leer el archivo CSV
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            data = [row for row in reader]

        return JSONResponse(content=data)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo CSV: {str(e)}")