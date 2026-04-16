"""
predict_prophet_gare.py
========================
Extension de predict_load_model.py pour les modèles Prophet par gare.
Peut être utilisé seul ou intégré dans l'API existante.

Nouvelles routes :
  GET  /prophet/gare/{gare}              → JSON prédictions avec bruit
  GET  /prophet/gare/{gare}/plot         → PNG interactif
  GET  /prophet/gare/{gare}/components   → PNG décomposition Prophet
  GET  /prophet/all_gares                → toutes les gares en un appel
  GET  /prophet/health                   → statut des modèles Prophet
"""

import os, sys, io, json, base64, warnings, logging, threading
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("prophet_api")

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from app.database import engine

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
GARES      = ["Cotonou-Nord", "Allada", "Houegbo", "Porto-Novo", "Epke", "Parakou"]
NOISE_STD  = 0.8
FUTURE_HRS = 168

GARE_COLORS = {
    "Cotonou-Nord": "#2563EB", "Allada"  : "#059669",
    "Houegbo"     : "#D97706", "Porto-Novo": "#DC2626",
    "Epke"        : "#7C3AED", "Parakou" : "#0891B2",
}


def _safe(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


# ════════════════════════════════════════════════════════════════════════════
# REGISTRE PROPHET — chargé une seule fois
# ════════════════════════════════════════════════════════════════════════════
_prophet: dict = {}
_p_ready = False
_p_error = None


def _load_prophet_all():
    global _p_ready, _p_error
    try:
        log.info("=== Chargement modèles Prophet par gare ===")
        for gare in GARES:
            key         = _safe(gare)
            model_path  = os.path.join(MODELS_DIR, f"prophet_gare_{key}.pkl")
            params_path = os.path.join(MODELS_DIR, f"prophet_gare_{key}_params.json")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modèle manquant : {model_path}\n"
                                        f"Lance d'abord : python train_prophet_gare.py")

            with open(params_path) as f:
                params = json.load(f)

            _prophet[gare] = {
                "model" : joblib.load(model_path),
                "params": params,
                "hourly": None,     # rempli depuis DB
            }
            log.info(f"  ✓ {gare}")

        # Charger les données horaires depuis PostgreSQL (GROUP BY côté SQL)
        for gare in GARES:
            sql = f"""
                SELECT DATE_TRUNC('hour', date_heure) AS ds, COUNT(*) AS y
                FROM fact_transactions
                WHERE gare = '{gare}'
                GROUP BY 1 ORDER BY 1
            """
            df = pd.read_sql(sql, engine)
            df["ds"] = pd.to_datetime(df["ds"])
            df.set_index("ds", inplace=True)
            full_idx = pd.date_range(df.index.min(), df.index.max(), freq="h")
            hourly   = df.reindex(full_idx, fill_value=0)
            hourly.index.name = "ds"
            _prophet[gare]["hourly"] = hourly

        _p_ready = True
        log.info(f"=== {len(_prophet)} modèles Prophet prêts ===")

    except Exception as e:
        _p_error = str(e)
        log.error(f"Chargement Prophet échoué : {e}")


# ════════════════════════════════════════════════════════════════════════════
# PRÉDICTION PROPHET AVEC BRUIT GAUSSIEN
# ════════════════════════════════════════════════════════════════════════════
def _prophet_predict(
    gare: str,
    future_hours: int = FUTURE_HRS,
    noise_std: float  = NOISE_STD,
) -> dict:
    """
    Génère future_hours prédictions Prophet avec bruit gaussien.

    Réalisme :
      - Prophet prédit la tendance propre + intervalles de confiance
      - Bruit N(0, σ_adapté) ajouté sur chaque point futur
        σ_adapté = noise_std × std_historique × 0.15
      - Clip [0, cap] avec cap = 130% du max historique
    """
    entry    = _prophet[gare]
    model    = entry["model"]
    hourly   = entry["hourly"]
    params   = entry["params"]

    # Données complètes pour reconstruire le contexte
    agg = hourly.reset_index().rename(columns={"ds": "ds", "y": "y"})
    agg.columns = ["ds", "y"]

    # Prédiction Prophet
    future   = model.make_future_dataframe(periods=future_hours, freq="h")
    forecast = model.predict(future)

    # Séparation futur
    cutoff    = agg["ds"].max()
    future_fc = forecast[forecast["ds"] > cutoff].copy().reset_index(drop=True)

    # Bruit adapté à l'échelle de la gare
    y_std        = float(agg["y"].std())
    noise_scaled = noise_std * y_std * 0.15
    cap          = float(agg["y"].max()) * 1.3

    noise = np.random.normal(0, noise_scaled, len(future_fc))

    future_fc["yhat_final"]       = np.clip(future_fc["yhat"]       + noise, 0, cap)
    future_fc["yhat_lower_final"] = np.clip(future_fc["yhat_lower"] + noise, 0, cap)
    future_fc["yhat_upper_final"] = np.clip(future_fc["yhat_upper"] + noise, 0, cap)

    return {
        "gare"       : gare,
        "forecast"   : forecast,
        "future_fc"  : future_fc,
        "agg"        : agg,
        "params"     : params,
        "cutoff"     : cutoff,
        "noise_used" : round(noise_scaled, 4),
    }


def _format_predictions(result: dict) -> list:
    fc = result["future_fc"]
    return [
        {
            "hour"       : i,
            "date"       : str(row["ds"]),
            "predicted"  : round(float(row["yhat_final"]), 2),
            "lower_95"   : round(float(row["yhat_lower_final"]), 2),
            "upper_95"   : round(float(row["yhat_upper_final"]), 2),
        }
        for i, row in fc.iterrows()
    ]


def _build_forecast_png(gare: str, result: dict) -> bytes:
    agg       = result["agg"]
    future_fc = result["future_fc"]
    cutoff    = result["cutoff"]
    color     = GARE_COLORS.get(gare, "#2563EB")
    hist_hrs  = min(4 * 7 * 24, len(agg))

    fig, ax = plt.subplots(figsize=(16, 7))

    # Historique
    last_hist = agg.tail(hist_hrs)
    ax.plot(last_hist["ds"], last_hist["y"],
            label="Historique réel", color=color, lw=1.5, alpha=0.75)

    # Prévision avec bruit
    ax.plot(future_fc["ds"], future_fc["yhat_final"],
            label=f"Prévision Prophet (σ={NOISE_STD})",
            color="#DC2626", lw=2.5, ls="--")

    # IC 95%
    ax.fill_between(future_fc["ds"],
                    future_fc["yhat_lower_final"],
                    future_fc["yhat_upper_final"],
                    alpha=0.18, color="#DC2626", label="IC 95%")

    # Séparation
    ax.axvline(x=cutoff, color="gray", ls=":", lw=1.5, label="Début prévision")
    ax.axvspan(future_fc["ds"].iloc[0], future_fc["ds"].iloc[-1],
               alpha=0.05, color="#DC2626")

    # Métriques dans le titre
    m = result["params"]["metrics"]
    ax.set_title(
        f"Gare {gare} — Prévision Prophet 7 jours  "
        f"(MAE={m['mae']:.2f} | MAPE={m['mape']:.1f}%)",
        fontsize=13, fontweight="bold",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Véhicules / heure")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0); data = buf.read(); plt.close(fig)
    return data


def _build_components_png(gare: str, result: dict) -> bytes:
    model    = _prophet[gare]["model"]
    forecast = result["forecast"]
    color    = GARE_COLORS.get(gare, "#2563EB")

    comp_fig  = model.plot_components(forecast)
    comp_axes = comp_fig.get_axes()

    fig, axes = plt.subplots(len(comp_axes), 1, figsize=(16, 5 * len(comp_axes)))
    if len(comp_axes) == 1:
        axes = [axes]

    component_colors = [color, "#7C3AED", "#059669", "#D97706"]
    for i, (src_ax, dst_ax) in enumerate(zip(comp_axes, axes)):
        c = component_colors[i % len(component_colors)]
        for line in src_ax.get_lines():
            dst_ax.plot(line.get_xdata(), line.get_ydata(), color=c, lw=2)
        for coll in src_ax.collections:
            try:
                paths = coll.get_paths()
                if paths:
                    verts = paths[0].vertices
                    dst_ax.fill_between(verts[:, 0], verts[:, 1],
                                        alpha=0.2, color=c)
            except Exception:
                pass
        dst_ax.set_title(src_ax.get_title() or src_ax.get_ylabel(), fontweight="bold")
        dst_ax.set_xlabel("Date")
        dst_ax.grid(True, alpha=0.3)

    plt.close(comp_fig)
    fig.suptitle(f"Décomposition Prophet — Gare {gare}", fontsize=13, fontweight="bold")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0); data = buf.read(); plt.close(fig)
    return data


# ════════════════════════════════════════════════════════════════════════════
# FASTAPI
# ════════════════════════════════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_load_prophet_all, daemon=True).start()
    yield


app = FastAPI(
    title="Prophet Gare Prediction API",
    version="1.0.0",
    lifespan=lifespan,
)


def _chk():
    if _p_error:
        raise HTTPException(503, detail=f"Erreur chargement : {_p_error}")
    if not _p_ready:
        raise HTTPException(503, detail="Modèles en cours de chargement — réessaie dans 30s")

def _chk_gare(gare: str):
    if gare not in GARES:
        raise HTTPException(400, detail=f"Gare invalide. Valides : {GARES}")


# ── Info ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    return {
        "ready" : _p_ready,
        "gares" : GARES,
        "routes": {
            "GET /prophet/gare/{gare}"           : "Prédictions JSON",
            "GET /prophet/gare/{gare}/plot"       : "PNG forecast",
            "GET /prophet/gare/{gare}/components" : "PNG décomposition",
            "GET /prophet/all_gares"              : "Toutes les gares",
            "GET /prophet/health"                 : "Statut modèles",
        }
    }


@app.get("/prophet/health", tags=["Info"])
def prophet_health():
    return {
        "ready"  : _p_ready,
        "error"  : _p_error,
        "models" : [
            {
                "gare"   : g,
                "loaded" : g in _prophet,
                "hours"  : len(_prophet[g]["hourly"]) if g in _prophet and _prophet[g]["hourly"] is not None else 0,
                "metrics": _prophet[g]["params"]["metrics"] if g in _prophet else {},
            }
            for g in GARES
        ]
    }


# ── Prédictions JSON ──────────────────────────────────────────────────────
@app.get("/prophet/gare/{gare}", tags=["Prédictions"])
def predict_gare(
    gare        : str,
    future_hours: int   = FUTURE_HRS,
    noise_std   : float = NOISE_STD,
):
    _chk(); _chk_gare(gare)
    result = _prophet_predict(gare, future_hours, noise_std)
    fc     = result["future_fc"]
    m      = result["params"]["metrics"]
    return {
        "status"     : "success",
        "gare"       : gare,
        "noise_used" : result["noise_used"],
        "metadata"   : {
            "future_hours"    : future_hours,
            "noise_std"       : noise_std,
            "mean_prediction" : round(float(fc["yhat_final"].mean()), 2),
            "min_prediction"  : round(float(fc["yhat_final"].min()), 2),
            "max_prediction"  : round(float(fc["yhat_final"].max()), 2),
            "train_metrics"   : m,
        },
        "predictions": _format_predictions(result),
    }


# ── Plot PNG ──────────────────────────────────────────────────────────────
@app.get("/prophet/gare/{gare}/plot", tags=["Plots"])
def plot_gare(gare: str, future_hours: int = FUTURE_HRS, noise_std: float = NOISE_STD):
    _chk(); _chk_gare(gare)
    result = _prophet_predict(gare, future_hours, noise_std)
    return Response(_build_forecast_png(gare, result), media_type="image/png")


# ── Décomposition ─────────────────────────────────────────────────────────
@app.get("/prophet/gare/{gare}/components", tags=["Plots"])
def components_gare(gare: str, future_hours: int = FUTURE_HRS):
    _chk(); _chk_gare(gare)
    result = _prophet_predict(gare, future_hours)
    return Response(_build_components_png(gare, result), media_type="image/png")


# ── Toutes les gares ──────────────────────────────────────────────────────
@app.get("/prophet/all_gares", tags=["Prédictions"])
def all_gares(future_hours: int = FUTURE_HRS, noise_std: float = NOISE_STD):
    _chk()
    out = {"status": "success", "gares": {}}
    for gare in GARES:
        result = _prophet_predict(gare, future_hours, noise_std)
        fc     = result["future_fc"]
        out["gares"][gare] = {
            "mean"       : round(float(fc["yhat_final"].mean()), 2),
            "max"        : round(float(fc["yhat_final"].max()), 2),
            "metrics"    : result["params"]["metrics"],
            "predictions": _format_predictions(result),
        }
    return out
