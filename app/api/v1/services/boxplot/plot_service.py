import os 
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from app.api.v1.services.boxplot.summary_plot_service import generate_summary_statistics
from app.api.v1.services.data_service import load_csv
from app.api.v1.classes.boxplot_input import UserBoxplotInput

GRAPH_DIR = os.path.join(os.getcwd(), "Grafics", "Boxplot_Sleep")
os.makedirs(GRAPH_DIR, exist_ok=True)


""" Columna por defecto para el eje Y """

Default_Eje_Y = "Sleep_Hours_Per_Night"

def generate_boxplot_with_user_point(user_input: UserBoxplotInput, output_path: str, df: pd.DataFrame = None) -> str:
    
    """ Vamos a generar la gráfica por defecto si el usuario no proporciona un DataFrame """
    if df is None:
        df = load_csv()

    """ Asignación del eje Y basado en la entrada del usuario o el valor por defecto """
    eje_y = user_input.eje_y if user_input.eje_y else Default_Eje_Y
    
    """ Validammos que las columnas necesarias existan en el DataFrame """

    if eje_y not in df.columns:
        raise ValueError(f"La columna '{eje_y}' no existe en el DataFrame.")
    
    """ Creamos la gráfica general """

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), sharey=True)
    sns.boxplot(data=df, y=eje_y, ax=axes[0])
    axes[0].set_title(f"Distribución general")

    """ Creamos la gráfica del usuario """
    sns.boxplot(data=df, y=eje_y, ax=axes[1])
    axes[1].axhline(user_input.sleep_time, color='red', linestyle='--', label='Tu punto')
    axes[1].set_title("Tu punto en contexto")
    axes[1].legend()

    """ Configuraciones generales de la gráfica """
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

def generate_boxplot_with_user_point_and_summary(user_input: UserBoxplotInput, df: pd.DataFrame = None) -> str:

    if df is None:
        df = load_csv()

    """ Asignación del eje Y basado en la entrada del usuario o el valor por defecto """
    eje_y = user_input.eje_y if user_input.eje_y else Default_Eje_Y

    """ Validamos que las columnas necesarias existan en el DataFrame """
    if eje_y not in df.columns:
        raise ValueError(f"La columna '{eje_y}' no existe en el DataFrame.")

    """ Generamos el resumen de estadísticas """
    resumen = generate_summary_statistics(user_input.sleep_time, eje_y, df)

    return resumen