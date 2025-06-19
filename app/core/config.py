import os

class Settings:
    PROJECT_NAME: str = "FastAPI App"
    VERSION: str = "1.0.0"
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")

settings = Settings()
