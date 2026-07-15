"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from backend.api.routes import forecast, health, simulate, report, insights

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
app.include_router(forecast.router, prefix="/api/v1", tags=["Forecast"])
app.include_router(simulate.router, prefix="/api/v1", tags=["Budget Simulator"])
app.include_router(report.router, prefix="/api/v1", tags=["Report"])
app.include_router(insights.router, prefix="/api/v1", tags=["AI Insights"])

