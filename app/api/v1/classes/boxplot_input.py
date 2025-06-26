from pydantic import BaseModel
from typing import Optional

class UserBoxplotInput(BaseModel):
    eje_y: Optional[str] = None
    user_value: float