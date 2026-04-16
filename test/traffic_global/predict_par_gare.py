import sys
import os
import warnings
import logging
import threading
import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from contextlib import asynccontextmanager
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import load_model


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lstm_gare_api")
tf.get_logger().setLevel("ERROR")


# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models_par_gare")
LOOKBACK = 168
NOISE_STD = 0.8

# État global avec cache des modèles
_state = {
    "models": {},      # {station: model}
    "scalers": {},     # {station: scaler}
    "ready": False,
    "error": None,
    "stations": [],
}


def _load_all_models():
    """Charge tous les modèles de stations au démarrage"""
    log.info("=== Chargement des modèles par gare ===")
    try:
        if not os.path.exists(MODELS_DIR):
            raise FileNotFoundError(f"Dossier modèles introuvable: {MODELS_DIR}")
        
        # Trouver toutes les stations
        stations = set()
        for file in os.listdir(MODELS_DIR):
            if file.endswith("_model.keras"):
                station = file.replace("_model.keras", "")
                stations.add(station)
        
        stations = sorted(stations)
        _state["stations"] = stations
        log.info(f"Stations trouvées: {stations}")
        
        # Charger modèles et scalers
        for station in stations:
            model_path = os.path.join(MODELS_DIR, f"{station}_model.keras")
            scaler_path = os.path.join(MODELS_DIR, f"{station}_scaler.pkl")
            
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                log.info(f"Chargement: {station}...")
                _state["models"][station] = load_model(model_path)
                _state["scalers"][station] = joblib.load(scaler_path)
                
                # Warm-up
                dummy = np.zeros((1, LOOKBACK, 1), dtype=np.float32)
                _state["models"][station].predict(dummy, verbose=0)
                log.info(f"  ✓ {station} chargé")
            else:
                log.warning(f"  ✗ Modèle ou scaler manquant pour {station}")
        
        _state["ready"] = True
        log.info(f"=== {len(_state['models'])} modèles chargés ===")
        
    except Exception as e:
        _state["error"] = str(e)
        log.error(f"Erreur chargement: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=_load_all_models, daemon=True)
    t.start()
    yield
    log.info("Arrêt API")


app = FastAPI(
    title="LSTM Traffic Prediction by Station API",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_ready():
    if _state["error"]:
        raise HTTPException(status_code=503, detail=f"Erreur chargement: {_state['error']}")
    if not _state["ready"]:
        raise HTTPException(status_code=503, detail="Modèles en cours de chargement...")


def _require_station(station: str):
    if station not in _state["models"]:
        raise HTTPException(
            status_code=400,
            detail=f"Station '{station}' inconnue. Disponibles: {_state['stations']}"
        )


@tf.function(reduce_retracing=True)
def _predict_step(model, x):
    return model(x, training=False)


def _predict_for_station(station: str, future_hours: int = 168, noise_std: float = NOISE_STD) -> dict:
    """Génère prédictions pour une station"""
    start_time = time.time()
    
    model = _state["models"][station]
    scaler = _state["scalers"][station]
    
    # Données historiques fictives (dans la pratique, viendrait de DB)
    # Génération de séquence d'initialisation
    last_vals = np.random.uniform(10, 50, size=LOOKBACK).reshape(-1, 1)
    input_seq = scaler.transform(last_vals).astype(np.float32)
    
    future_clean = np.empty(future_hours, dtype=np.float32)
    
    for i in range(future_hours):
        x = input_seq.reshape(1, LOOKBACK, 1)
        pred_clean = float(_predict_step(model, tf.constant(x))[0, 0])
        future_clean[i] = pred_clean
        
        # Bruit gaussien
        pred_noisy = float(np.clip(pred_clean + np.random.normal(0, noise_std), 0.0, 1.0))
        input_seq = np.append(input_seq[1:], [[pred_noisy]], axis=0)
    
    # Dénormalisation
    preds = scaler.inverse_transform(future_clean.reshape(-1, 1)).flatten()
    preds = np.clip(preds, 0, None)
    
    # Dates futures
    base_date = pd.Timestamp.now()
    future_dates = pd.date_range(start=base_date, periods=future_hours, freq="h")
    
    processing_time = time.time() - start_time
    
    return {
        "predictions": preds,
        "dates": future_dates,
        "mean": float(preds.mean()),
        "min": float(preds.min()),
        "max": float(preds.max()),
        "processing_time_seconds": round(processing_time, 3),
    }


# ════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ════════════════════════════════════════════════════════════════════════════
class PredictionRequest(BaseModel):
    future_hours: int = 168
    noise_std: float = 0.8


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════
@app.get("/", tags=["Info"])
def root():
    return {
        "name": "LSTM Traffic Prediction by Station API",
        "ready": _state["ready"],
        "stations": _state["stations"],
        "endpoints": {
            "GET  /health": "Statut serveur",
            "GET  /stations": "Liste des stations",
            "GET  /predict/plot": "Plot interactif pour une station",
            "POST /predict/json": "Prédictions JSON pour une station",
        },
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "ready": _state["ready"],
        "error": _state["error"],
        "stations_loaded": len(_state["models"]),
        "total_stations": len(_state["stations"]),
        "lookback_hours": LOOKBACK,
    }


@app.get("/stations", tags=["Info"])
def list_stations():
    return {
        "stations": _state["stations"],
        "count": len(_state["stations"]),
    }


@app.get("/predict/plot", tags=["Predictions"])
def predict_plot_by_station(
    station: str = Query(..., description="Nom de la gare"),
    future_hours: int = Query(168, description="Nombre d'heures à prédire"),
    noise_std: float = Query(0.8, description="Écart-type du bruit"),
):
    """Retourne un plot interactif Plotly pour une station"""
    _require_ready()
    _require_station(station)
    
    result = _predict_for_station(station, future_hours, noise_std)
    
    # Créer figure Plotly
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=result["dates"],
        y=result["predictions"],
        mode='lines',
        name=f'Prévision {station}',
        line=dict(color='#2563EB', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(37, 99, 235, 0.1)',
        hovertemplate='<b>Prévision</b><br>Date: %{x|%Y-%m-%d %H:%M}<br>Véhicules: %{y:.0f}<extra></extra>',
    ))
    
    fig.update_layout(
        title={
            'text': f'<b>Prévisions Trafic - Gare: {station}</b><br><sub>Horizon: {future_hours}h</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        xaxis_title='<b>Date & Heure</b>',
        yaxis_title='<b>Véhicules par heure</b>',
        template='plotly_white',
        hovermode='x unified',
        height=600,
        margin=dict(l=80, r=60, t=120, b=80),
    )
    
    # Wrapper HTML professionnel
    html_content = f"""
    <!DOCTYPE html>
    <html lang="fr">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Prévisions Trafic - {station}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                overflow: hidden;
            }}
            .header {{
                background: linear-gradient(135deg, #2563EB 0%, #1e40af 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                font-size: 28px;
                margin-bottom: 10px;
            }}
            .content {{
                padding: 30px;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }}
            .stat-item {{
                background: #f3f4f6;
                padding: 15px;
                border-radius: 8px;
                text-align: center;
                border-left: 4px solid #2563EB;
            }}
            .stat-label {{
                font-size: 12px;
                color: #6b7280;
                text-transform: uppercase;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #1f2937;
                margin-top: 8px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>📍 Trafic Prédits - Gare {station}</h1>
                <p>Prévision LSTM pour {future_hours} heures</p>
            </div>
            <div class="content">
                {fig.to_html(include_plotlyjs='cdn', config={'responsive': True, 'displayModeBar': True})}
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-label">Moyenne</div>
                        <div class="stat-value">{result['mean']:.0f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Minimum</div>
                        <div class="stat-value">{result['min']:.0f}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Maximum</div>
                        <div class="stat-value">{result['max']:.0f}</div>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)


@app.post("/predict/json", tags=["Predictions"])
def predict_json_by_station(
    station: str = Query(..., description="Nom de la gare"),
    request: PredictionRequest = None,
):
    """Retourne les prédictions en JSON pour une station"""
    _require_ready()
    _require_station(station)
    
    if request is None:
        request = PredictionRequest()
    
    result = _predict_for_station(station, request.future_hours, request.noise_std)
    
    return {
        "status": "success",
        "station": station,
        "metadata": {
            "future_hours": request.future_hours,
            "noise_std": request.noise_std,
            "mean_prediction": round(result["mean"]),
            "min_prediction": round(result["min"]),
            "max_prediction": round(result["max"]),
            "processing_time_seconds": result["processing_time_seconds"],
        },
        "predictions": [
            {
                "hour": i,
                "date": str(d),
                "predicted_vehicles": round(float(p)),
            }
            for i, (d, p) in enumerate(zip(result["dates"], result["predictions"]))
        ],
    }


@app.post("/predict/multi", tags=["Predictions"])
def predict_multi_stations(
    stations: list[str] = Query(..., description="Liste des gares"),
    future_hours: int = Query(168),
):
    """Prédictions pour plusieurs stations à la fois"""
    _require_ready()
    
    results = {}
    for station in stations:
        try:
            _require_station(station)
            result = _predict_for_station(station, future_hours, NOISE_STD)
            results[station] = {
                "status": "success",
                "mean": round(result["mean"]),
                "min": round(result["min"]),
                "max": round(result["max"]),
            }
        except HTTPException as e:
            results[station] = {"status": "error", "detail": str(e.detail)}
    
    return results



if __name__ == "__main__":
    uvicorn.run("predict_par_gare:app", host="0.0.0.0", port=8001, reload=False)
