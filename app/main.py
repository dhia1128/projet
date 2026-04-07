from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from sqlalchemy import text
from pathlib import Path
from app.plots_plotly import (
    plot_transactions_by_payment,
    plot_montant_distribution,
    plot_top_gares_ca,
    plot_montant_by_vehicle,
    plot_daily_trend,
    plot_monthly_revenue_by_payment,
    plot_transactions_by_hour,
    plot_revenue_by_weekday,
    get_data
)
from app.database import get_engine
from app.prevision_v_heurewithpophet import visualise
from app.prophet_forecasts import plot_classe, plot_gare, plot_global
from app.traffic_stats_service import get_traffic_stats


app = FastAPI(title="TollXpress Dashboard - Plotly", version="1.0")

# Include prediction endpoints

# Get the templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"

@app.get("/health")
async def health_check():
    """
    Check if the database connection is working.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
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


@app.get("/plot/monthly_payment", response_class=HTMLResponse)
async def monthly_payment_plot():
    return plot_monthly_revenue_by_payment()


@app.get("/plot/hourly_transactions", response_class=HTMLResponse)
async def hourly_transactions_plot():
    return plot_transactions_by_hour()


@app.get("/plot/weekday_revenue", response_class=HTMLResponse)
async def weekday_revenue_plot():
    return plot_revenue_by_weekday()

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    with open(TEMPLATES_DIR / "dashboard.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(html_content)


@app.get("/traffic/stats", response_class=HTMLResponse)
def traffic_stats_page():
    with open(TEMPLATES_DIR / "traffic_stats.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(html_content)


@app.get("/api/traffic/stats", response_class=JSONResponse)
def traffic_stats_data():
    return JSONResponse(content=get_traffic_stats())


@app.get("/prevision", response_class=HTMLResponse)
async def prevision_plot():
    return visualise()


@app.get("/plot/classe", response_class=HTMLResponse)
def route_classe():  
    return plot_classe()

@app.get("/plot/gare", response_class=HTMLResponse)
def route_gare():
    return plot_gare()

@app.get("/plot/global", response_class=HTMLResponse)
def route_global():
    return plot_global()
    
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)