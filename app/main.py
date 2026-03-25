from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path
from app.plots_plotly import (
    plot_transactions_by_payment,
    plot_montant_distribution,
    plot_top_gares_ca,
    plot_montant_by_vehicle,
    plot_daily_trend
)

app = FastAPI(title="TollXpress Dashboard - Plotly", version="1.0")

# Get the templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
    