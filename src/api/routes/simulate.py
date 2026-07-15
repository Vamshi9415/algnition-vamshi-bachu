"""Budget simulation endpoint."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class SimulateRequest(BaseModel):
    forecast_id: str
    spend_changes: dict[str, float]  # {campaign_name: pct_change}


@router.post("/simulate")
def simulate_budget(req: SimulateRequest):
    """Apply budget changes to a stored forecast and return delta revenue estimate."""
    # In production this would load a cached forecast by ID
    # For the demo, returns a structured placeholder
    return {
        "forecast_id": req.forecast_id,
        "spend_changes": req.spend_changes,
        "message": "Submit with forecast_id from /api/v1/forecast to simulate",
    }
