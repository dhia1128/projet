import sys
import os
import io
import base64
import warnings
import logging
import threading
import time
from pathlib import Path
from fastapi.responses import FileResponse, HTMLResponse
import  uvicorn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # backend non-interactif — évite les crashes serveur
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import joblib
from contextlib import asynccontextmanager
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import text
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TensorFlow au boot
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lstm_api")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel


# ─── Imports internes ────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.database import engine, get_engine
from app.suivi_service import (
    plot_transactions_by_payment,
    plot_montant_distribution,
    plot_top_gares_ca,
    plot_montant_by_vehicle,
    plot_monthly_revenue_by_payment,
    plot_transactions_by_hour,
    plot_revenue_by_weekday,
    get_data    
)
from app.analyse_service import analyse_global, analyse_par_classe, analyse_par_gare
from app.Statstiques_service import get_traffic_stats


# ════════════════════════════════════════════════════════════════════════════
# CONFIG — doit être identique à l'entraînement
# ════════════════════════════════════════════════════════════════════════════
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "lstm_keras_best.keras")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")   # sauvegardé à l'entraînement
LOOKBACK    = 168    # ← CORRIGÉ : identique à previsionvperhour.py (était 23)
NOISE_STD   = 0.8    # identique à l'entraînement

# Get the templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"

# ════════════════════════════════════════════════════════════════════════════
# SINGLETON — chargé une seule fois, jamais rechargé
# ════════════════════════════════════════════════════════════════════════════
_state = {
    "model"  : None,
    "scaler" : None,
    "hourly" : None,
    "ready"  : False,
    "error"  : None,
}


def _load_everything():
    """
    Chargement unique au démarrage :
      - modèle Keras depuis disque
      - scaler depuis joblib (ou reconstruction depuis DB)
      - données horaires agrégées depuis PostgreSQL (GROUP BY côté SQL)
    Cette fonction tourne dans un thread background pour ne pas bloquer.
    """
    log.info("=== Chargement modèle + données ===")
    try:
        # 1. Modèle ─────────────────────────────────────────────────────────
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
        log.info(f"Chargement modèle : {MODEL_PATH}")
        _state["model"] = load_model(MODEL_PATH)
        # Warm-up : un seul appel pour que TF compile le graphe maintenant
        dummy = np.zeros((1, LOOKBACK, 1), dtype=np.float32)
        _state["model"].predict(dummy, verbose=0)
        log.info("Modèle chargé et compilé")

        # 2. Scaler ─────────────────────────────────────────────────────────
        if os.path.exists(SCALER_PATH):
            _state["scaler"] = joblib.load(SCALER_PATH)
            log.info(f"Scaler chargé depuis {SCALER_PATH}")
        else:
            # Fallback : reconstruction depuis les données DB
            log.warning(f"{SCALER_PATH} introuvable — reconstruction depuis DB")
            _state["scaler"] = None   # sera fitté après chargement hourly

        # 3. Données horaires — GROUP BY côté PostgreSQL (300k→730 lignes) ──
        sql = """
            SELECT
                DATE_TRUNC('hour', date_heure) AS ds,
                COUNT(*)                        AS nombre_voitures
            FROM fact_transactions
            GROUP BY DATE_TRUNC('hour', date_heure)
            ORDER BY ds
        """
        df = pd.read_sql(sql, engine)
        df["ds"] = pd.to_datetime(df["ds"])
        df.set_index("ds", inplace=True)

        # Compléter les heures manquantes
        full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
        hourly   = df.reindex(full_idx, fill_value=0)
        hourly.index.name = "ds"
        _state["hourly"] = hourly
        log.info(f"Données chargées : {len(hourly)} heures")

        # 4. Scaler fallback : fitté sur TOUT le dataset (pas seulement recent)
        if _state["scaler"] is None:
            _state["scaler"] = MinMaxScaler(feature_range=(0, 1))
            _state["scaler"].fit(hourly[["nombre_voitures"]])
            log.warning("Scaler reconstruit depuis DB — pensez à sauvegarder avec joblib")

        _state["ready"] = True
        log.info("=== Prêt — temps de réponse API < 200ms ===")

    except Exception as e:
        _state["error"] = str(e)
        log.error(f"Chargement échoué : {e}")


# ════════════════════════════════════════════════════════════════════════════
# LIFESPAN — remplace @app.on_event("startup") déprécié
# ════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lancement en background pour ne pas bloquer le démarrage
    t = threading.Thread(target=_load_everything, daemon=True)
    t.start()
    yield
    log.info("Arrêt API")


app = FastAPI(
    title="TollXpress Dashboard - Unified API (LSTM + Analytics)",
    version="2.1.0",
    lifespan=lifespan,
)


# ════════════════════════════════════════════════════════════════════════════
# HELPER — vérification état
# ════════════════════════════════════════════════════════════════════════════
def _require_ready():
    if _state["error"]:
        raise HTTPException(status_code=503, detail=f"Erreur chargement : {_state['error']}")
    if not _state["ready"]:
        raise HTTPException(
            status_code=503,
            detail="Modèle en cours de chargement — réessaie dans 30 secondes"
        )


# ════════════════════════════════════════════════════════════════════════════
# OPTIMISATION CENTRALE — prédictions vectorisées (pas de boucle .predict())
# ════════════════════════════════════════════════════════════════════════════
def _predict_fast(future_hours: int = 168, noise_std: float = NOISE_STD) -> dict:
    """
    Génère `future_hours` prédictions en utilisant le singleton chargé.

    OPTIMISATION CLEF :
    Au lieu de appeler model.predict() 168 fois (168 × ~120ms = 20s),
    on utilise tf.function pour compiler le graphe une seule fois et
    on prédit heure par heure avec le graphe compilé (~2ms par appel).

    noise_std appliqué uniquement sur la valeur réinjectée (pas sur le plot).
    """
    start_time = time.time()
    
    model  = _state["model"]
    scaler = _state["scaler"]
    hourly = _state["hourly"]

    # Séquence de départ : les LOOKBACK dernières heures connues
    last_vals = hourly["nombre_voitures"].values[-LOOKBACK:].reshape(-1, 1)
    input_seq = scaler.transform(last_vals).astype(np.float32)  # (168, 1)

    future_clean = np.empty(future_hours, dtype=np.float32)

    # tf.function compilé une seule fois — ~2ms par appel au lieu de 120ms
    @tf.function(reduce_retracing=True)
    def _step(x):
        return model(x, training=False)

    for i in range(future_hours):
        x = input_seq.reshape(1, LOOKBACK, 1)
        pred_clean  = float(_step(tf.constant(x))[0, 0])
        future_clean[i] = pred_clean

        # Bruit gaussien sur la valeur réinjectée seulement
        pred_noisy = float(np.clip(pred_clean + np.random.normal(0, noise_std), 0.0, 1.0))
        input_seq  = np.append(input_seq[1:], [[pred_noisy]], axis=0)

    # Dénormalisation
    preds = scaler.inverse_transform(future_clean.reshape(-1, 1)).flatten()
    preds = np.clip(preds, 0, None)

    last_ts      = hourly.index[-1]
    future_dates = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=future_hours, freq="h"
    )

    processing_time = time.time() - start_time

    return {
        "predictions" : preds,
        "dates"       : future_dates,
        "mean"        : float(preds.mean()),
        "min"         : float(preds.min()),
        "max"         : float(preds.max()),
        "processing_time_seconds": round(processing_time, 3),
    }


def _build_plot_plotly(future_hours: int, noise_std: float) -> tuple:
    """Génère un plot Plotly interactif en HTML avec timing."""
    start_time = time.time()
    
    result    = _predict_fast(future_hours, noise_std)
    hourly    = _state["hourly"]
    hist_hrs  = min(4 * 7 * 24, len(hourly))

    plot_start = time.time()
    
    # Historique réel
    hist_dates = hourly.index[-hist_hrs:]
    hist_values = hourly["nombre_voitures"].tail(hist_hrs).values
    
    # Prédictions
    pred_dates = result["dates"]
    pred_values = result["predictions"]
    
    # Créer la figure Plotly
    fig = go.Figure()
    
    # Ajouter la courbe historique
    fig.add_trace(go.Scatter(
        x=hist_dates,
        y=hist_values,
        name="Historique réel",
        mode="lines",
        line=dict(color="#2563EB", width=2),
        hovertemplate="<b>Historique</b><br>Date: %{x}<br>Véhicules: %{y:.0f}<extra></extra>"
    ))
    
    # Ajouter la courbe de prédiction
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=pred_values,
        name=f"Prévision LSTM ({future_hours}h)",
        mode="lines",
        line=dict(color="#DC2626", width=2, dash="dash"),
        hovertemplate="<b>Prédiction</b><br>Date: %{x}<br>Véhicules: %{y:.0f}<extra></extra>"
    ))
    
    # Zone de prédiction en arrière-plan
    fig.add_vrect(
        x0=pred_dates[0], x1=pred_dates[-1],
        fillcolor="#DC2626", opacity=0.1, layer="below", line_width=0,
    )
    
    # Ligne verticale à la limite entre histoire et prédiction
    fig.add_vline(
        x=hourly.index[-1],
        line_dash="dot", line_color="gray", line_width=1.5,
        annotation_text="Limite", annotation_position="top right"
    )
    
    # Mise en page
    fig.update_layout(
        title=dict(
            text=f"<b>Prédiction LSTM — {future_hours}h</b> (Bruit σ={noise_std})",
            font=dict(size=16)
        ),
        xaxis_title="Date",
        yaxis_title="Véhicules / heure",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        showlegend=True,
        legend=dict(
            x=0.02, y=0.98,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="lightgray"),
    )
    
    # Convertir en HTML
    html_str = fig.to_html(include_plotlyjs="cdn", div_id="lstm_plot")
    
    plot_time = time.time() - plot_start
    total_time = time.time() - start_time
    
    # Add timing information to result
    result["plot_generation_time_seconds"] = round(plot_time, 3)
    result["total_processing_time_seconds"] = round(total_time, 3)
    
    return html_str, result


def _build_plot_png(future_hours: int, noise_std: float) -> tuple:
    """Génère le plot et retourne les bytes PNG avec timing (legacy)."""
    start_time = time.time()
    
    result    = _predict_fast(future_hours, noise_std)
    hourly    = _state["hourly"]
    hist_hrs  = min(4 * 7 * 24, len(hourly))

    plot_start = time.time()
    
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(
        hourly.index[-hist_hrs:],
        hourly["nombre_voitures"].tail(hist_hrs),
        label="Historique réel", color="#2563EB", linewidth=1.5, alpha=0.8,
    )
    ax.plot(
        result["dates"], result["predictions"],
        label=f"Prévision LSTM ({future_hours}h)",
        color="#DC2626", linewidth=2, linestyle="--",
    )
    ax.axvspan(result["dates"][0], result["dates"][-1], alpha=0.08, color="#DC2626")
    ax.axvline(x=hourly.index[-1], color="gray", linestyle=":", linewidth=1.5)
    ax.set_title(f"LSTM — Prévision {future_hours}h (σ bruit={noise_std})",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Véhicules / heure")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    buf.seek(0)
    data = buf.read()
    plt.close(fig)
    
    plot_time = time.time() - plot_start
    total_time = time.time() - start_time
    
    # Add timing information to result
    result["plot_generation_time_seconds"] = round(plot_time, 3)
    result["total_processing_time_seconds"] = round(total_time, 3)
    
    return data, result


# ════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ════════════════════════════════════════════════════════════════════════════
class PredictionRequest(BaseModel):
    future_hours : int   = 168    # 7 jours
    noise_std    : float = 0.8    # identique à l'entraînement


# ════════════════════════════════════════════════════════════════════════════
# ROUTES
# ════════════════════════════════════════════════════════════════════════════
@app.get("/", tags=["Info"])
def root():
    return {
        "name"    : "TollXpress Dashboard - Unified API v2.1",
        "ready"   : _state["ready"],
        "lookback": LOOKBACK,
        "endpoints": {
            "Dashboard": {
                "GET  /home"               : "Home page",
                "GET  /dashboard"          : "Main dashboard",
            },
            "Analytics & Visualization": {
                "GET  /plot/paiement"      : "Payment methods plot",
                "GET  /plot/montant"       : "Amount distribution plot",
                "GET  /plot/ca_gare"       : "Revenue by station plot",
                "GET  /plot/vehicule"      : "Amount by vehicle plot",
                "GET  /plot/monthly_payment" : "Monthly revenue plot",
                "GET  /plot/hourly_transactions" : "Hourly transactions plot",
                "GET  /plot/weekday_revenue" : "Revenue by weekday plot",
            },
            "Analysis APIs": {
                "GET  /api/analyse/classe" : "Class analysis JSON",
                "GET  /api/analyse/gare"   : "Station analysis JSON",
                "GET  /api/analyse/global" : "Global analysis JSON",
                "GET  /analyse/classe"     : "Class analysis UI",
                "GET  /analyse/gare"       : "Station analysis UI",
                "GET  /analyse/global"     : "Global analysis UI",
            },
            "Traffic Stats": {
                "GET  /traffic/stats"      : "Traffic stats page",
                "GET  /api/traffic/stats"  : "Traffic stats JSON",
            },
            "LSTM Predictions": {
                "GET  /health"             : "System health status",
                "POST /predict/json"       : "Predictions JSON",
                "GET  /predict/plot"       : "Plot PNG direct",
                "GET  /predict/plotly"     : "Interactive Plotly Dashboard",
                "GET  /prevision"          : "Unified Prevision (LSTM + Class Trends)",
            },
            "Class Trends": {
                "GET  /class_trends"       : "Class trends data",
                "GET  /prevision/class_trends" : "Class trends API",
                "GET  /class_trends_dashboard" : "Class trends dashboard",
            },
        },
    }


@app.get("/health", tags=["Info"])
def health():
    """Health check endpoint — includes DB connection status"""
    db_status = "disconnected"
    db_error = None
    try:
        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_error = str(e)
    
    return {
        "ready"       : _state["ready"],
        "error"       : _state["error"],
        "model_loaded": _state["model"] is not None,
        "data_hours"  : len(_state["hourly"]) if _state["hourly"] is not None else 0,
        "lookback"    : LOOKBACK,
        "database"    : "PostgreSQL",
        "connected"   : db_status == "connected",
        "db_status"   : db_status,
        "db_error"    : db_error,
        "processing_time_seconds": _state.get("processing_time_seconds", None),
    }


@app.post("/predict/json", tags=["Predictions"])
def predict_json(request: PredictionRequest):
    """Prédictions JSON — < 5s grâce au singleton et tf.function."""
    _require_ready()
    result = _predict_fast(request.future_hours, request.noise_std)
    return {
        "status": "success",
        "metadata": {
            "future_hours"    : request.future_hours,
            "noise_std"       : request.noise_std,
            "lookback_hours"  : LOOKBACK,
            "mean_prediction" : round(result["mean"]),
            "min_prediction"  : round(result["min"]),
            "max_prediction"  : round(result["max"]),
            "processing_time_seconds": result["processing_time_seconds"],
        },
        "predictions": [
            {
                "hour"              : i,
                "date"              : str(d),
                "predicted_vehicles": round(float(p)),
            }
            for i, (d, p) in enumerate(zip(result["dates"], result["predictions"]))
        ],
    }


@app.get("/predict/plot", tags=["Predictions"])
def predict_plot(future_hours: int = 168, noise_std: float = NOISE_STD):
    """Retourne un PNG directement — ouvrable dans le navigateur."""
    _require_ready()
    start_time = time.time()
    png_bytes, result = _build_plot_png(future_hours, noise_std)
    total_time = time.time() - start_time
    
    log.info(f"Plot generation completed in {result['total_processing_time_seconds']}s")
    
    return Response(
        content=png_bytes, 
        media_type="image/png",
        headers={
            "X-Processing-Time-Seconds": str(result["total_processing_time_seconds"]),
            "X-Prediction-Time-Seconds": str(result["processing_time_seconds"]),
            "X-Plot-Generation-Time-Seconds": str(result["plot_generation_time_seconds"]),
        }
    )


@app.get("/predict/plotly", response_class=HTMLResponse, tags=["Predictions"])
def predict_plotly():
    """Serve interactive Plotly prediction dashboard"""
    _require_ready()
    with open(TEMPLATES_DIR / "predict_plotly.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(html_content)





# DASHBOARD ENDPOINTS — from main.py


@app.get("/home", response_class=HTMLResponse, tags=["Dashboard"])
async def home():
    with open(TEMPLATES_DIR / "home.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(html_content)


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
def dashboard():
    with open(TEMPLATES_DIR / "dashboard.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(html_content)



# VISUALIZATION ENDPOINTS — from main.py


@app.get("/plot/paiement", response_class=HTMLResponse, tags=["Visualization"])
async def paiement_plot():
    return plot_transactions_by_payment()


@app.get("/plot/montant", response_class=HTMLResponse, tags=["Visualization"])
async def montant_plot():
    return plot_montant_distribution()


@app.get("/plot/ca_gare", response_class=HTMLResponse, tags=["Visualization"])
async def ca_gare_plot():
    return plot_top_gares_ca()


@app.get("/plot/vehicule", response_class=HTMLResponse, tags=["Visualization"])
async def vehicule_plot():
    return plot_montant_by_vehicle()




@app.get("/plot/monthly_payment", response_class=HTMLResponse, tags=["Visualization"])
async def monthly_payment_plot():
    return plot_monthly_revenue_by_payment()


@app.get("/plot/hourly_transactions", response_class=HTMLResponse, tags=["Visualization"])
async def hourly_transactions_plot():
    return plot_transactions_by_hour()


@app.get("/plot/weekday_revenue", response_class=HTMLResponse, tags=["Visualization"])
async def weekday_revenue_plot():
    return plot_revenue_by_weekday()



# TRAFFIC STATS ENDPOINTS — from main.py


@app.get("/traffic/stats", response_class=HTMLResponse, tags=["Traffic Stats"])
def traffic_stats_page():
    with open(TEMPLATES_DIR / "traffic_stats.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(html_content)


@app.get("/api/traffic/stats", response_class=JSONResponse, tags=["Traffic Stats"])
def traffic_stats_data():
    return JSONResponse(content=get_traffic_stats())


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS API ENDPOINTS — from main.py
# ════════════════════════════════════════════════════════════════════════════

@app.get("/api/analyse/classe", response_class=JSONResponse, tags=["Analysis"])
def api_analyse_classe():
    """Get class analysis data as JSON"""
    try:
        data = analyse_par_classe()
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/api/analyse/gare", response_class=JSONResponse, tags=["Analysis"])
def api_analyse_gare():
    """Get station analysis data as JSON"""
    try:
        data = analyse_par_gare()
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.get("/api/analyse/global", response_class=JSONResponse, tags=["Analysis"])
def api_analyse_global():
    """Get global analysis data as JSON"""
    try:
        data = analyse_global()
        return JSONResponse(content=data)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


# ════════════════════════════════════════════════════════════════════════════
# ANALYSIS UI ENDPOINTS — from main.py
# ════════════════════════════════════════════════════════════════════════════

@app.get("/analyse/classe", response_class=HTMLResponse, tags=["Analysis"])
def route_classe():
    """Display class analysis with interactive UI"""
    with open(TEMPLATES_DIR / "classe_analyse.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)


@app.get("/analyse/gare", response_class=HTMLResponse, tags=["Analysis"])
def route_gare():
    """Display station analysis with interactive UI"""
    with open(TEMPLATES_DIR / "gare_analyse.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)


@app.get("/analyse/global", response_class=HTMLResponse, tags=["Analysis"])
def route_global():
    """Display global analysis with interactive UI"""
    with open(TEMPLATES_DIR / "global_analyse.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(html)









if __name__ == "__main__":
    uvicorn.run("predict_load_model:app", host="0.0.0.0", port=8000, reload=False)

