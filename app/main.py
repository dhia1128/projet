from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from pathlib import Path
from app.plots_plotly import (
    plot_transactions_by_payment,
    plot_montant_distribution,
    plot_top_gares_ca,
    plot_montant_by_vehicle,
    plot_daily_trend
)
from app.database import engine
from app.prediction_service import generate_predictions

app = FastAPI(title="TollXpress Dashboard - Plotly", version="1.0")

# Get the templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"

@app.get("/health")
async def health_check():
    """
    Check if the database connection is working.
    """
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "database": f"disconnected - {str(e)}", "error": str(e)}

@app.get("/home", response_class=HTMLResponse)
async def home():
    with open(TEMPLATES_DIR / "home.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(html_content)


@app.get("/plot/paiement", response_class=HTMLResponse)
async def paiement_plot():
    return plot_transactions_by_payment()

@app.get("/plot/montant", response_class=HTMLResponse)
async def montant_plot():
    return plot_montant_distribution()

@app.get("/plot/ca_gare", response_class=HTMLResponse)
async def ca_gare_plot():
    return plot_top_gares_ca()

@app.get("/plot/vehicule", response_class=HTMLResponse)
async def vehicule_plot():
    return plot_montant_by_vehicle()

@app.get("/plot/daily", response_class=HTMLResponse)
async def daily_plot():
    return plot_daily_trend()

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    with open(TEMPLATES_DIR / "dashboard.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(html_content)


# ==================== PREDICTION ENDPOINTS ====================

@app.get("/predictions/api")
async def get_predictions_api(hours_ahead: int = Query(24, ge=1, le=168)):
    """
    Get LSTM predictions as JSON
    hours_ahead: Number of hours to predict ahead (1-168)
    """
    result = generate_predictions(engine, hours_ahead=hours_ahead)
    return JSONResponse(result)


@app.get("/predictions", response_class=HTMLResponse)
async def predictions_dashboard(hours_ahead: int = Query(24, ge=1, le=168)):
    """
    Serve the predictions dashboard HTML
    """
    try:
        with open(TEMPLATES_DIR / "predictions.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(html_content)
    except FileNotFoundError:
        return HTMLResponse("""
        <html>
        <body style="font-family: Arial; padding: 20px;">
            <h1> Error</h1>
            <p>predictions.html template not found</p>
            <p><a href="/home">← Back to Home</a></p>
        </body>
        </html>
        """, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
    