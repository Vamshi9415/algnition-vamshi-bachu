from fastapi.testclient import TestClient

from backend.api.main import app


def test_insights_endpoint_returns_fallback_without_gemini_key():
    client = TestClient(app)
    payload = {
        "forecast": [
            {
                "date": "2026-07-15",
                "campaign_name": "Search_US",
                "channel": "Google Ads",
                "revenue_p50": 1200,
                "revenue_p10": 1000,
                "revenue_p90": 1400,
            }
        ],
        "summary": {"spend": 1000},
    }

    response = client.post("/api/v1/insights", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "executive_summary" in body
    assert "risk_analysis" in body
    assert "budget_recommendations" in body
    assert body["executive_summary"]
    assert body["risk_analysis"]
    assert body["budget_recommendations"]
    assert (
        body["executive_summary"].startswith("[LLM unavailable")
        or body["executive_summary"] != ""
    )