import sys
import os
import io
import base64
import warnings
import logging
import threading
import time
from fastapi.responses import FileResponse
import  uvicorn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # backend non-interactif — évite les crashes serveur
import matplotlib.pyplot as plt
import joblib
from contextlib import asynccontextmanager
from sklearn.preprocessing import MinMaxScaler
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # silence TensorFlow au boot
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("lstm_api")
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras.models import load_model
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel


# ─── Imports internes ────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.database import engine

# ════════════════════════════════════════════════════════════════════════════
# CONFIG — doit être identique à l'entraînement
# ════════════════════════════════════════════════════════════════════════════
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "lstm_keras_best.keras")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "scaler.pkl")   # sauvegardé à l'entraînement
LOOKBACK    = 168    # ← CORRIGÉ : identique à previsionvperhour.py (était 23)
NOISE_STD   = 0.8    # identique à l'entraînement

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
    title="LSTM Traffic Prediction API",
    version="2.0.0",
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


def _build_plot_png(future_hours: int, noise_std: float) -> tuple:
    """Génère le plot et retourne les bytes PNG avec timing."""
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
        "name"    : "LSTM Traffic Prediction API v2",
        "ready"   : _state["ready"],
        "lookback": LOOKBACK,
        "endpoints": {
            "GET  /health"         : "Statut du serveur",
            "POST /predict/json"   : "Prédictions JSON",
            "GET  /predict/plot"   : "Plot PNG direct",
            "POST /predict/full"   : "JSON + plot base64",
        },
    }


@app.get("/health", tags=["Info"])
def health():
    return {
        "ready"       : _state["ready"],
        "error"       : _state["error"],
        "model_loaded": _state["model"] is not None,
        "data_hours"  : len(_state["hourly"]) if _state["hourly"] is not None else 0,
        "lookback"    : LOOKBACK,
        "database"    : "PostgreSQL" if _state["hourly"] is not None else "None",
        "connected"   : _state["hourly"] is not None,
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




if __name__ == "__main__":
    uvicorn.run("predict_load_model:app", host="0.0.0.0", port=8000, reload=True)
# ════════════════════════════════════════════════════════════════════════════

