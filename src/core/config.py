from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Synthetic Data Generator API"
    DATA_DIR: Path = Path("data")
    TEMP_DIR: Path = DATA_DIR / "temp"
    
    class Config:
        case_sensitive = True

settings = Settings() 