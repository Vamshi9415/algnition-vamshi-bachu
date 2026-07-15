"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import forecast, upload, simulate, health

app = FastAPI(
    title="AIgnition Forecast Studio",
    description="AI-Assisted Probabilistic Revenue Forecasting Platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(forecast.router, prefix="/api")
app.include_router(simulate.router, prefix="/api")
