"""Report download endpoint."""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()


@router.get("/report", response_class=HTMLResponse)
def download_report():
    """Returns an HTML forecast report (generated after /forecast is called)."""
    html = """
    <html><body>
    <h1>AIgnition Forecast Report</h1>
    <p>Run <code>POST /api/v1/forecast</code> first to generate a report.</p>
    </body></html>
    """
    return HTMLResponse(content=html)

