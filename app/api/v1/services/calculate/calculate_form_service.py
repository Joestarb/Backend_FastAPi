import os
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression 
from app.api.v1.classes.form_input import FormInputUser
from app.api.v1.services.data_service import load_csv

def estimate_mental_health_and_addiction_score(data: FormInputUser):
    try:
        df = load_csv()

        """ Zona de variables """
        X = df[['Sleep_Hours_Per_Night', 'Avg_Daily_Usage_Hours']]
        Y = df['Mental_Health_Score']
        max_usage_hours = 10

        """ Esta variable se calcula haciendo uso de la media de las horas de uso diario """
        normalized_usage_hours = min(data.Avg_Daily_Usage_Hours, max_usage_hours) / max_usage_hours * 10

        """ Zona de entrenamiento del modelo """
        model = LinearRegression()
        model.fit(X, Y)

        """ Zona de predicción """
        input_data = np.array([[data.Sleep_Hours_Per_Night, data.Avg_Daily_Usage_Hours]])
        predicted_mental_score = float(model.predict(input_data)[0])

        """ Esta variable se calcula en base a la predicción del modelo """
        addicted_score = (normalized_usage_hours + (10 - predicted_mental_score)) / 2

        """ Calculamos el rendimiento académico """
        X_academic = df[['Mental_Health_Score', 'Addicted_Score']]
        Y_academic = df['Affects_Academic_Performance'].apply(lambda x: 1 if str(x).lower() in ['yes', 'true', '1'] else 0)

        """ Entrenamos el modelo de regresión lineal para el rendimiento académico """
        model_academic = LogisticRegression()
        model_academic.fit(X_academic, Y_academic)

        """ Realizamos la predicción del rendimiento académico """
        prediction_clf = model_academic.predict([[predicted_mental_score, addicted_score]])[0]
        prob = model_academic.predict_proba([[predicted_mental_score, addicted_score]])[0][1]

        performance_impact = True if prediction_clf == 1 else False

        """ Las variables de salida, son el estado de salud mental y el puntaje de adicción """

        return {
            "Mental_Health_Score": round(predicted_mental_score, 2),
            "Mental_Health_Interpretation": (
                "Alto riesgo" if predicted_mental_score < 4 else
                "Riesgo moderado" if predicted_mental_score < 7 else
                "Bajo riesgo"
            ),
            "Addiction_Score": round(addicted_score, 2),
            "Addiction_Level": (
                "Alta adicción" if addicted_score > 7 else
                "Adicción moderada" if addicted_score >= 5 else
                "Baja adicción"
            ),
            "Affects_Academic_Performance": performance_impact,
            "Academic_Performance_Probability": round(prob, 2),
        }

    except Exception as e:
        raise ValueError(f"Error al estimar el puntaje de salud mental: {str(e)}") from e
