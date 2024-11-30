from fastapi import FastAPI
from src.api.routes import synthetic_data
from src.core.config import settings
from src.core.logging import setup_logging

# Setup logging
setup_logging()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Include routers
app.include_router(
    synthetic_data.router,
    prefix=f"{settings.API_V1_STR}/synthetic-data",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=4) 