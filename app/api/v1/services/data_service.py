import pandas as pd
import os

CSV_DIR = os.path.join(os.getcwd(), "CSV")
CSV_PATH = os.path.join(CSV_DIR, "Prueba.csv")

""" Aquí se define el servicio para cargar el documento CSV """

def load_csv() -> pd.DataFrame:
    try:
        df = pd.read_csv(CSV_PATH)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo CSV en la ruta: {CSV_PATH}")

