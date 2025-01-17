import os
from typing import *

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()
class Settings(BaseSettings):
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "")
    BACKEND_CORS_ORIGINS: List[AnyStr] = ["*"]
    DB_PATH: str = os.getenv("DB_PATH", "")
    
settings = Settings()