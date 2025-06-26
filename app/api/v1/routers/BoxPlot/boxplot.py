import os
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.api.v1.classes.boxplot_input import UserBoxplotInput
from app.api.v1.services.boxplot.plot_service import generate_boxplot_with_user_point, generate_boxplot_with_user_point_and_summary

router = APIRouter()
GRAPH_SLEEP_VS_SCREEN = os.path.join(os.getcwd(), "Grafics", "Boxplot_Sleep", "sleep_vs_screen.png")


@router.post("/boxplot/sleep_vs_screen")
def get_plot(user_input: UserBoxplotInput):
    try: 
        path = generate_boxplot_with_user_point(user_input, GRAPH_SLEEP_VS_SCREEN)
        return FileResponse(path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al generar la gráfica: " + str(e))

@router.post("/boxplot/summary_statistics")
def get_summary_statistics(user_input: UserBoxplotInput):
    try:
        summary = generate_boxplot_with_user_point_and_summary(user_input)
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error al generar las estadísticas: " + str(e))