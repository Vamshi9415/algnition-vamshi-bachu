"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from backend.api import state
from backend.api.routes import forecast, health, simulate, report, insights, upload

app = FastAPI(
    title="AIgnition Forecast Studio API",
    description="Probabilistic revenue forecasting for digital marketing agencies.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
app.include_router(forecast.router, prefix="/api/v1", tags=["Forecast"])
app.include_router(simulate.router, prefix="/api/v1", tags=["Budget Simulator"])
app.include_router(report.router, prefix="/api/v1", tags=["Report"])
app.include_router(insights.router, prefix="/api/v1", tags=["AI Insights"])


@app.on_event("startup")
def _warm_model_cache() -> None:
    """Load the pre-trained model + historical dataset once at boot, so the first
    forecast request isn't the one paying the load cost."""
    try:
        state.get_state()
    except RuntimeError as exc:
        logger.warning(f"Model not pre-loaded at startup: {exc}")

