import pandas as pd

def generate_summary_statistics(user_value: float, columna: str, df: pd.DataFrame) -> dict:
    serie = df[columna].dropna()
    descripcion = serie.describe()

    q1 = serie.quantile(0.25)
    q2 = serie.median()
    q3 = serie.quantile(0.75)

    if user_value < q1:
        position = "por debajo del 25%"
    elif q1 <= user_value < q2:
        position = "Entre el 25% y el 50% del rango intercuartil"
    elif q2 <= user_value < q3:
        position = "Entre el 50% y el 75% del rango intercuartil"
    else:
        position = "por encima del 75%"

    return {
        "min": round(descripcion["min"], 2),
        "q1": round(q1, 2),
        "mediana": round(q2, 2),
        "q3": round(q3, 2),
        "max": round(descripcion["max"], 2),
        "usuario": user_value,
        "cuartil_usuario": position
    }



