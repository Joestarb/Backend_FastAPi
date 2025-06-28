import os
from fastapi import APIRouter
from app.api.v1.classes.form_input import FormInputUser
from app.api.v1.services.calculate.calculate_form_service import estimate_mental_health_and_addiction_score

router = APIRouter()

@router.post("/prediction/user_form")
def prediction_user_form(data: FormInputUser):
    result = estimate_mental_health_and_addiction_score(data)
    return result

