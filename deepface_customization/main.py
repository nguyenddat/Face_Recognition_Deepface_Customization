import uvicorn
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from test_api.api import recognition
from test_api.core import config

def get_application() -> FastAPI:
    application = FastAPI()
    
    application.add_middleware(
        CORSMiddleware,
        allow_origins = [str(origin) for origin in config.settings.BACKEND_CORS_ORIGINS],
        allow_credentials = True,
        allow_methods = ["*"],
        allow_headers = ["*"]
    )
    
    application.include_router(recognition.router, tags = ["verify"])
    return application

app = get_application()

if __name__ == "__main__":
    uvicorn.run("main:app", host = "0.0.0.0", port = 8000, reload = True)
