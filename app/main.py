from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
from sqlalchemy import text
from pathlib import Path
from app.database import get_engine
from app.suivi_service import (
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
from app.analyse_service import analyse_global, analyse_par_classe, analyse_par_gare
from app.Statstiques_service import get_traffic_stats
from app.repartionparclasse_service import _load_classes_data, _calculate_hourly_patterns, _calculate_daily_patterns


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




# ====================== ANALYSIS API ENDPOINTS ======================

@app.get("/api/analyse/classe", response_class=JSONResponse)
def api_analyse_classe():
    """Get class analysis data as JSON"""
    try:
        data = analyse_par_classe()
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/api/analyse/gare", response_class=JSONResponse)
def api_analyse_gare():
    """Get station analysis data as JSON"""
    try:
        data = analyse_par_gare()
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/api/analyse/global", response_class=JSONResponse)
def api_analyse_global():
    """Get global analysis data as JSON"""
    try:
        data = analyse_global()
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# ====================== ANALYSIS UI ENDPOINTS ======================

@app.get("/analyse/classe", response_class=HTMLResponse)
def route_classe():
    """Display class analysis with interactive UI"""
    with open(TEMPLATES_DIR / "classe_analyse.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)


@app.get("/analyse/gare", response_class=HTMLResponse)
def route_gare():
    """Display station analysis with interactive UI"""
    with open(TEMPLATES_DIR / "gare_analyse.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)


@app.get("/analyse/global", response_class=HTMLResponse)
def route_global():
    """Display global analysis with interactive UI"""
    with open(TEMPLATES_DIR / "global_analyse.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)


@app.get("/class_trends", response_class=JSONResponse)
def get_data_api():
    data = get_data()
    return JSONResponse(content=data.to_dict(orient="records"))

@app.get("/prevision/class_trends", response_class=JSONResponse)
def class_trends_api():
    """API endpoint that returns class trends data as JSON"""
    loaded_data = _load_classes_data()
    hourly_patterns = _calculate_hourly_patterns(loaded_data)
    daily_patterns = _calculate_daily_patterns(loaded_data)
    return JSONResponse(content={"hourly_patterns": hourly_patterns.to_dict(), "daily_patterns": daily_patterns.to_dict()})

@app.get("/class_trends_dashboard", response_class=HTMLResponse)
def class_trends_dashboard():
    """Serve the class trends visualization dashboard"""
    with open(TEMPLATES_DIR / "class_trends.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(html_content)



@app.get("/prevision", response_class=HTMLResponse)
def prevision_page():
    with open(TEMPLATES_DIR / "prevision.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(html_content)



if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)