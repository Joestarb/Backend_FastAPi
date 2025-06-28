from pydantic import BaseModel
from typing import Optional

class FormInputUser(BaseModel):
    Age: int = None
    Avg_Daily_Usage_Hours: float = None
    #Affects_Academic_Performance: bool = None
    Sleep_Hours_Per_Night: float = None
    Conflicts_Over_Social_Media: int = None
    Academic_Level_Graduate: bool = None
    Academic_Level_High_School: bool = None
    Academic_Level_Undergraduate: bool = None
    Most_Used_Platform_Facebook: bool = None
    Most_Used_Platform_Instagram: bool = None
    Most_Used_Platform_KakaoTalk: bool = None
    Most_Used_Platform_LINE: bool = None
    Most_Used_Platform_LinkedIn: bool = None
    Most_Used_Platform_Snapchat: bool = None
    Most_Used_Platform_TikTok: bool = None
    Most_Used_Platform_Twitter: bool = None
    Most_Used_Platform_VKontakte: bool = None
    Most_Used_Platform_WeChat: bool = None
    Most_Used_Platform_WhatsApp: bool = None
    Most_Used_Platform_YouTube: bool = None
    Relationship_Status_Complicated: bool = None
    Relationship_Status_In_Relationship: bool = None
    Relationship_Status_Single: bool = None
    Gender_Female: bool = None
    Gender_Male: bool = None




