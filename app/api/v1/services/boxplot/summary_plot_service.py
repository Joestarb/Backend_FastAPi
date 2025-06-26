import pandas as pd

def generate_summary_statistics(user_value: float, columna: str, df: pd.DataFrame) -> dict:
    serie = df[columna].dropna()
    descripcion = serie.describe()

    min_val = round(descripcion["min"], 2)
    max_val = round(descripcion["max"], 2)
    q1 = round(serie.quantile(0.25), 2)
    q2 = round(serie.median(), 2)
    q3 = round(serie.quantile(0.75), 2)
    promedio = round(descripcion["mean"], 2)
    desviacion = round(descripcion["std"], 2)

    if user_value < q1:
        posicion = "Estás en el grupo con los valores más bajos."
        interpretacion = (
            f"Tu valor ({user_value}) está entre los más bajos de todos los registrados en '{columna}'. "
            f"La mayoría de las personas tienen un valor mayor que el tuyo, y solo un pequeño porcentaje tiene resultados parecidos o más bajos. "
            f"Esto puede ser una señal de alerta dependiendo del contexto de esta variable."
        )
        recomendacion = (
            "Sería buena idea analizar este resultado con calma. Si esta variable refleja salud, bienestar o desempeño, podrías beneficiarte de apoyo, asesoría o pequeños cambios positivos."
        )

    elif q1 <= user_value < q2:
        posicion = "Estás un poco por debajo del promedio."
        interpretacion = (
            f"Tu resultado ({user_value}) está por debajo del promedio general ({promedio}), "
            f"pero no se encuentra entre los más bajos. Hay muchas personas con resultados similares, aunque aún hay espacio para mejorar."
        )
        recomendacion = (
            "Es un buen momento para hacer pequeños ajustes que te ayuden a seguir avanzando. Estás cerca del promedio, y con constancia podrías superarlo fácilmente."
        )

    elif q2 <= user_value < q3:
        posicion = "Estás por encima del promedio."
        interpretacion = (
            f"Tu valor ({user_value}) está por encima de la mayoría, lo cual es una buena señal. "
            f"Muchas personas tienen resultados más bajos que tú, lo que indica un desempeño o estado positivo."
        )
        recomendacion = (
            "¡Bien hecho! Trata de mantener o incluso mejorar ese nivel. Vas por buen camino."
        )

    else:
        posicion = "Estás entre los valores más altos."
        interpretacion = (
            f"Tu resultado ({user_value}) está entre los más altos registrados para '{columna}'. "
            f"Eso te coloca dentro del grupo con mejor desempeño o estado."
        )
        recomendacion = (
            "¡Excelente! Estás en un nivel destacado. Si esta variable refleja algo positivo, estás dando un gran ejemplo."
        )

    return {
        "columna_analizada": columna,
        "min": min_val,
        "q1": q1,
        "mediana": q2,
        "q3": q3,
        "max": max_val,
        "media": promedio,
        "desviacion_estandar": desviacion,
        "valor_usuario": user_value,
        "posicion_usuario": posicion,
        "interpretacion": interpretacion,
        "recomendacion": recomendacion
    }
