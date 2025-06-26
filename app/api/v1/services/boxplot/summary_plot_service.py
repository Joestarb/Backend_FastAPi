import pandas as pd

Col_saludable = "Mental_Health_Score"
Col_Edad = "Age"
Col_problematica_1 = "Conflicts_Over_Social_Media"
Col_problematica_2 = "Addicted_Score"
Col_problematica_3 = "Avg_Daily_Usage_Hours"
Col_Descanso = "Sleep_Hours_Per_Night"

VARIABLES_PROBLEMATICAS = {
    Col_problematica_1,
    Col_problematica_2,
    Col_problematica_3
}

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

    es_problematica = columna in VARIABLES_PROBLEMATICAS

    # Lógica invertida si la variable es problemática

    if es_problematica:
        if user_value <= 2:
            posicion = "Nivel muy bajo"
            interpretacion = (
                f"Tu resultado ({user_value}) indica que prácticamente no tienes conflictos, adicción o uso excesivo relacionado con '{columna}'. "
                "Esto refleja un excelente manejo en este aspecto."
            )
            recomendacion = "Mantén estos hábitos. El equilibrio digital y emocional es fundamental."

        elif user_value <= 4:
            posicion = "Nivel bajo"
            interpretacion = (
                f"Tu resultado ({user_value}) sugiere un nivel bajo de impacto en '{columna}'. "
                "Podrías experimentar pequeñas tensiones o hábitos poco frecuentes, pero no es preocupante."
            )
            recomendacion = "Sigue prestando atención y mantén tu autocontrol."

        elif user_value <= 6:
            posicion = "Nivel medio"
            interpretacion = (
                f"Tu resultado ({user_value}) está en un punto intermedio. "
                "Aunque no es alarmante, sí es un nivel donde vale la pena reflexionar sobre el impacto que esto puede tener."
            )
            recomendacion = "Evalúa tus rutinas y busca reducir gradualmente si notas efectos negativos."

        elif user_value <= 8:
            posicion = "Nivel alto"
            interpretacion = (
                f"Tu resultado ({user_value}) indica una tendencia considerable hacia el conflicto, la adicción o el uso intensivo en '{columna}'. "
                "Esto podría estar afectando tu bienestar, tus relaciones o tu rendimiento diario."
            )
            recomendacion = "Te convendría tomar acciones concretas: reducir tiempo, establecer límites, buscar apoyo o alternativas."

        else:
            posicion = "Nivel crítico"
            interpretacion = (
                f"Tu resultado ({user_value}) es muy alto. Esto sugiere una presencia intensa de problemas en '{columna}', lo cual es preocupante. "
                "Podría estar afectando significativamente tu vida emocional, social o física."
            )
            recomendacion = "Busca ayuda o guía profesional. Es importante actuar pronto para recuperar el equilibrio y tu bienestar."

    else:
        # Variables saludables (como salud mental)
        if user_value < q1:
            posicion = "Estás en el grupo con los valores más bajos."
            interpretacion = (
                f"Tu valor ({user_value}) está entre los más bajos registrados en '{columna}'. "
                "Esto sugiere que podrías estar enfrentando desafíos en este aspecto."
            )
            recomendacion = "Considera buscar apoyo o realizar pequeños cambios positivos que mejoren tu bienestar."

        elif q1 <= user_value < q2:
            posicion = "Estás un poco por debajo del promedio."
            interpretacion = (
                f"Tu valor ({user_value}) está por debajo del promedio general ({promedio}). "
                "Aunque no estás en el grupo más bajo, todavía hay oportunidad de mejora."
            )
            recomendacion = "Con pequeños pasos podrías acercarte a un mejor nivel. La constancia hace la diferencia."

        elif q2 <= user_value < q3:
            posicion = "Estás por encima del promedio."
            interpretacion = (
                f"Tu valor ({user_value}) supera al promedio general ({promedio}), lo cual es positivo. "
                "Refleja un estado o desempeño favorable en este aspecto."
            )
            recomendacion = "¡Muy bien! Sigue con esos hábitos y busca mantener o incluso mejorar tu nivel."

        else:
            posicion = "Estás entre los valores más altos."
            interpretacion = (
                f"Tu valor ({user_value}) es de los más altos en '{columna}', indicando un nivel sobresaliente."
            )
            recomendacion = "Excelente resultado. Es un ejemplo de bienestar y equilibrio. ¡Sigue así!"

    if columna == Col_Edad:
        if user_value < 14:
            posicion = "Muy joven (10–13 años)"
            interpretacion = (
                f"Tienes {user_value} años, lo que corresponde a una etapa muy temprana de desarrollo. "
                "Es un momento clave para construir hábitos positivos, explorar intereses y cuidar tu salud física y emocional."
            )
            recomendacion = "Aprovecha tu curiosidad y energía para aprender y mantener un estilo de vida saludable."

        elif user_value < 18:
            posicion = "Adolescente (14–17 años)"
            interpretacion = (
                f"Tienes {user_value} años, una etapa de muchos cambios y descubrimientos. "
                "Es común tener altibajos, pero también es una gran oportunidad para formar bases sólidas para tu futuro."
            )
            recomendacion = "Busca equilibrio entre tus estudios, tus pasatiempos y tu descanso. Rodéate de personas positivas."

        elif user_value < 22:
            posicion = "Joven adulto (18–21 años)"
            interpretacion = (
                f"Con {user_value} años, estás dando los primeros pasos hacia la vida adulta. "
                "Estás construyendo tu independencia y es un excelente momento para afianzar hábitos, metas y relaciones."
            )
            recomendacion = "Organiza tus prioridades, cuida tu salud mental y rodéate de apoyo positivo."

        elif user_value < 26:
            posicion = "Adulto joven (22–25 años)"
            interpretacion = (
                f"A los {user_value} años ya tienes más claridad sobre tus intereses y responsabilidades. "
                "Puede ser una etapa de alta exigencia, pero también de mucho crecimiento personal y profesional."
            )
            recomendacion = "Mantén el balance entre tu trabajo, tus vínculos y tu bienestar físico y emocional."

        else:
            posicion = "Adulto joven consolidado (26–30 años)"
            interpretacion = (
                f"Con {user_value} años, probablemente ya tengas una rutina definida. "
                "Es una buena etapa para revisar tus metas, mejorar tu bienestar general y compartir tu experiencia con otros."
            )
            recomendacion = "Refuerza lo que has aprendido, mantén tu salud al día y sigue cultivando tus relaciones y aspiraciones."

    if columna == Col_Descanso:
        diferencia = abs(user_value - 8)

        if diferencia < 0.5:
            posicion = "Óptimo"
            interpretacion = (
                f"Duermes {user_value} horas al día, lo cual es prácticamente ideal. "
                "Un buen descanso mejora la memoria, el estado de ánimo y la salud en general."
            )
            recomendacion = "Sigue manteniendo una rutina de sueño regular. Tu cuerpo y mente te lo agradecen."

        elif diferencia < 1.5:
            posicion = "Adecuado"
            interpretacion = (
                f"Duermes {user_value} horas al día, lo cual está bastante cerca del ideal. "
                "Es un rango saludable, aunque podrías notar una ligera diferencia si te acercas a 8 horas exactas."
            )
            recomendacion = "Intenta afinar tu rutina para alcanzar las 8 horas. Un ajuste pequeño puede marcar la diferencia."

        elif diferencia < 3:
            posicion = "Subóptimo"
            interpretacion = (
                f"Duermes {user_value} horas al día, lo que se aleja un poco del ideal. "
                "Podrías experimentar efectos leves como cansancio, falta de concentración o irritabilidad."
            )
            recomendacion = "Evalúa si puedes ajustar tu tiempo de descanso. Dormir mejor puede impactar positivamente muchas áreas."

        else:
            if user_value < 8:
                posicion = "Déficit de sueño"
                interpretacion = (
                    f"Duermes solo {user_value} horas al día, lo cual es considerablemente menos de lo recomendado. "
                    "Esto puede afectar tu salud física, mental y emocional."
                )
                recomendacion = "Busca maneras de priorizar tu descanso. El sueño insuficiente sostenido puede ser perjudicial."

            else:
                posicion = "Exceso de sueño"
                interpretacion = (
                    f"Duermes {user_value} horas al día, lo que supera ampliamente las recomendaciones. "
                    "Dormir demasiado también puede estar asociado con fatiga, bajo ánimo o condiciones médicas subyacentes."
                )
                recomendacion = "Observa cómo te sientes al despertar y durante el día. Si el exceso de sueño persiste, consulta a un profesional."

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
