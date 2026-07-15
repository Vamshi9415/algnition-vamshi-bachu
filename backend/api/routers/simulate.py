"""Budget simulation endpoint."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from backend.api.routers.upload import _canonical_cache
from ml_engine.budget.simulator import BudgetSimulator
from ml_engine.llm.insights import InsightGenerator

router = APIRouter(tags=["simulate"])
llm = InsightGenerator()


class SimulationRequest(BaseModel):
    channel: str
    spend_change_pct: float
    base_forecast_p50: float


@router.post("/simulate")
def simulate_budget(req: SimulationRequest):
    df = _canonical_cache.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No data uploaded.")

    simulator = BudgetSimulator()
    simulator.fit(df)
    result = simulator.simulate(req.channel, req.spend_change_pct, req.base_forecast_p50)
    recommendation = llm.generate_budget_recommendation([result])
    result["ai_recommendation"] = recommendation
    return JSONResponse(result)


@router.post("/simulate/all")
def simulate_all_channels(base_forecast_p50: float, spend_change_pct: float = 20.0):
    df = _canonical_cache.get("df")
    if df is None:
        raise HTTPException(status_code=400, detail="No data uploaded.")

    simulator = BudgetSimulator()
    simulator.fit(df)
    results = simulator.simulate_all_channels(spend_change_pct, base_forecast_p50)
    recommendation = llm.generate_budget_recommendation(results)
    return JSONResponse({"simulations": results, "ai_recommendation": recommendation})

