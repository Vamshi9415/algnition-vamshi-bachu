"""Budget simulation endpoint."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from src.api.routers.upload import _canonical_cache
from src.budget.simulator import BudgetSimulator
from src.llm.insights import LLMInsightGenerator

router = APIRouter(tags=["simulate"])
llm = LLMInsightGenerator()


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
