"""
train_prophet_gare.py
======================
Entraîne un modèle Prophet par gare avec des prédictions réalistes.

Réalisme obtenu par :
  1. Saisonnalité journalière + hebdomadaire + annuelle
  2. Bruit gaussien σ=0.8 appliqué sur la séquence de prédiction
  3. Clip [0, max_historique] pour éviter les valeurs absurdes
  4. changepoint_prior_scale=0.15 pour capturer les vraies tendances

Sauvegarde par gare dans models/ :
  prophet_gare_{safe_name}.pkl        ← modèle sérialisé
  prophet_gare_{safe_name}_params.json ← metadata + métriques

Run :
  python train_prophet_gare.py
"""

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════════
CSV_PATH   = "donnees_synthetiques_tollxpress_benin_2023-2024.csv"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

GARES        = ["Cotonou-Nord", "Allada", "Houegbo", "Porto-Novo", "Epke", "Parakou"]
FUTURE_HOURS = 168     # 7 jours = 168 heures
NOISE_STD    = 0.8     # même valeur que le LSTM
TRAIN_RATIO  = 0.8

GARE_COLORS = {
    "Cotonou-Nord": "#2563EB",
    "Allada"      : "#059669",
    "Houegbo"     : "#D97706",
    "Porto-Novo"  : "#DC2626",
    "Epke"        : "#7C3AED",
    "Parakou"     : "#0891B2",
}


def _safe(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


# ════════════════════════════════════════════════════════════════════════════
# CHARGEMENT & AGRÉGATION
# ════════════════════════════════════════════════════════════════════════════
def load_gare(df_raw: pd.DataFrame, gare: str) -> pd.DataFrame:
    """Retourne DataFrame Prophet {ds, y} horaire pour une gare."""
    mask = df_raw["Gare"] == gare
    agg  = (
        df_raw[mask]
        .groupby(df_raw[mask]["Date_Heure"].dt.floor("h"))
        .size()
        .reset_index(name="y")
        .rename(columns={"Date_Heure": "ds"})
    )
    # Compléter les heures manquantes
    full_idx = pd.date_range(agg["ds"].min(), agg["ds"].max(), freq="h")
    agg = (
        agg.set_index("ds")
           .reindex(full_idx, fill_value=0)
           .reset_index()
           .rename(columns={"index": "ds"})
    )
    return agg


# ════════════════════════════════════════════════════════════════════════════
# ENTRAÎNEMENT PROPHET
# ════════════════════════════════════════════════════════════════════════════
def train_prophet(agg: pd.DataFrame) -> Prophet:
    """
    Configure et entraîne Prophet pour des prédictions horaires réalistes.

    Choix :
    - seasonality_mode='additive'  → évite les valeurs négatives
    - daily + weekly + yearly      → capture toutes les saisonnalités du trafic
    - changepoint_prior_scale=0.15 → tendance flexible (détecte les vrais changements)
    - fourier_order élevé (daily=10, weekly=5) → courbes plus précises
    """
    model = Prophet(
        seasonality_mode="additive",
        changepoint_prior_scale=0.15,
        seasonality_prior_scale=10.0,
        interval_width=0.95,
        daily_seasonality=False,    # on la redéfinit manuellement pour contrôler fourier_order
        weekly_seasonality=False,
        yearly_seasonality=False,
    )

    # Saisonnalité journalière — haute précision (fourier_order=10)
    model.add_seasonality(
        name="daily",
        period=1,
        fourier_order=10,
        prior_scale=10.0,
    )
    # Saisonnalité hebdomadaire
    model.add_seasonality(
        name="weekly",
        period=7,
        fourier_order=5,
        prior_scale=10.0,
    )
    # Saisonnalité annuelle
    model.add_seasonality(
        name="yearly",
        period=365.25,
        fourier_order=8,
        prior_scale=10.0,
    )

    model.fit(agg)
    return model


# ════════════════════════════════════════════════════════════════════════════
# PRÉDICTION AVEC BRUIT GAUSSIEN — réalisme identique au LSTM
# ════════════════════════════════════════════════════════════════════════════
def predict_with_noise(
    model: Prophet,
    agg: pd.DataFrame,
    future_hours: int = FUTURE_HOURS,
    noise_std: float = NOISE_STD,
) -> pd.DataFrame:
    """
    Génère future_hours prédictions horaires avec bruit gaussien.

    Stratégie réalisme :
      - Prophet prédit la tendance + saisonnalité (valeurs "propres")
      - On ajoute bruit gaussien N(0, σ) sur chaque prédiction future
      - Clip [0, cap] pour rester dans des valeurs plausibles
      - σ est adapté à l'échelle de la cible (pas normalisé [0,1] comme LSTM)
    """
    future   = model.make_future_dataframe(periods=future_hours, freq="h")
    forecast = model.predict(future)

    # Séparer historique et futur
    cutoff    = agg["ds"].max()
    future_fc = forecast[forecast["ds"] > cutoff].copy()

    # Adapter le bruit à l'échelle réelle de la cible
    # σ_adapté = noise_std × std(y_historique)  ← plus réaliste que σ fixe
    y_std        = float(agg["y"].std())
    noise_scaled = noise_std * y_std * 0.15    # 15% de la std historique

    cap = float(agg["y"].max()) * 1.3          # plafond = 130% du max historique

    np.random.seed(None)   # seed aléatoire à chaque appel
    noise = np.random.normal(0, noise_scaled, len(future_fc))

    future_fc["yhat_noisy"]       = np.clip(future_fc["yhat"]       + noise, 0, cap)
    future_fc["yhat_lower_noisy"] = np.clip(future_fc["yhat_lower"] + noise, 0, cap)
    future_fc["yhat_upper_noisy"] = np.clip(future_fc["yhat_upper"] + noise, 0, cap)

    return forecast, future_fc


# ════════════════════════════════════════════════════════════════════════════
# MÉTRIQUES SUR LE TEST SET
# ════════════════════════════════════════════════════════════════════════════
def compute_metrics(model: Prophet, agg: pd.DataFrame) -> dict:
    """Évalue le modèle sur les 20% derniers points."""
    n_train = int(len(agg) * TRAIN_RATIO)
    test    = agg.iloc[n_train:].copy()

    forecast_test = model.predict(test[["ds"]])
    y_true = test["y"].values
    y_pred = np.clip(forecast_test["yhat"].values, 0, None)

    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100)

    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "mape": round(mape, 2)}


# ════════════════════════════════════════════════════════════════════════════
# PLOTS DE DIAGNOSTIC
# ════════════════════════════════════════════════════════════════════════════
def plot_diagnostic(gare: str, agg: pd.DataFrame, forecast: pd.DataFrame,
                    future_fc: pd.DataFrame, metrics: dict):
    color = GARE_COLORS.get(gare, "#2563EB")
    cutoff = agg["ds"].max()

    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.30)

    # (A) Forecast complet
    ax1 = fig.add_subplot(gs[0, :])
    last_90 = agg[agg["ds"] >= agg["ds"].max() - pd.Timedelta(days=90)]
    ax1.plot(last_90["ds"], last_90["y"],
             label="Réel (90j)", color=color, lw=1.5, alpha=0.7)
    ax1.plot(future_fc["ds"], future_fc["yhat_noisy"],
             label=f"Prévision {FUTURE_HOURS}h (bruit σ={NOISE_STD})",
             color="#DC2626", lw=2, ls="--")
    ax1.fill_between(future_fc["ds"],
                     future_fc["yhat_lower_noisy"],
                     future_fc["yhat_upper_noisy"],
                     alpha=0.15, color="#DC2626", label="IC 95%")
    ax1.axvline(cutoff, color="gray", ls=":", lw=1.5, label="Début prévision")
    ax1.set_title(f"Gare {gare} — Prévision Prophet 7 jours  "
                  f"(MAE={metrics['mae']:.2f} | MAPE={metrics['mape']:.1f}%)",
                  fontweight="bold")
    ax1.set_xlabel("Date"); ax1.set_ylabel("Véhicules / heure")
    ax1.legend(loc="upper left"); ax1.grid(True, alpha=0.3)

    # (B) Composantes Prophet
    comp_fig = model_obj.plot_components(forecast)
    comp_axes = comp_fig.get_axes()
    for j, comp_ax in enumerate(comp_axes[:2]):   # trend + daily
        ax_dst = fig.add_subplot(gs[1, j])
        for line in comp_ax.get_lines():
            ax_dst.plot(line.get_xdata(), line.get_ydata(),
                        color=color if j == 0 else "#7C3AED", lw=2)
        ax_dst.set_title(comp_ax.get_title() or comp_ax.get_ylabel(), fontweight="bold")
        ax_dst.set_xlabel("Date" if j == 0 else "Heure")
        ax_dst.set_ylabel("Effet")
        ax_dst.grid(True, alpha=0.3)
    plt.close(comp_fig)

    fig.suptitle(f"Prophet — Diagnostic complet · Gare {gare}", fontsize=14, fontweight="bold")
    plot_path = os.path.join(MODELS_DIR, f"prophet_gare_{_safe(gare)}_diagnostic.png")
    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot   → {plot_path}")


# ════════════════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE — entraîne et sauvegarde chaque gare
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time
    start = time.time()

    print(f"\n{'='*60}")
    print("  ENTRAÎNEMENT PROPHET — TRAFIC PAR GARE")
    print(f"{'='*60}")
    print(f"  CSV         : {CSV_PATH}")
    print(f"  Gares       : {GARES}")
    print(f"  Future hrs  : {FUTURE_HOURS} ({FUTURE_HOURS//24} jours)")
    print(f"  Bruit       : σ={NOISE_STD}\n")

    df_raw = pd.read_csv(CSV_PATH)
    df_raw["Date_Heure"] = pd.to_datetime(df_raw["Date_Heure"])

    all_metrics = {}

    for gare in GARES:
        print(f"\n{'─'*50}")
        print(f"  Gare : {gare}")
        print(f"{'─'*50}")

        # ── Données ───────────────────────────────────────────────────────
        agg = load_gare(df_raw, gare)
        n_train = int(len(agg) * TRAIN_RATIO)
        agg_train = agg.iloc[:n_train].copy()
        print(f"  Heures train : {len(agg_train)} | test : {len(agg) - n_train}")

        # ── Entraînement ──────────────────────────────────────────────────
        model_obj = train_prophet(agg_train)

        # ── Métriques ────────────────────────────────────────────────────
        metrics = compute_metrics(model_obj, agg)
        all_metrics[gare] = metrics
        print(f"  MAE={metrics['mae']:.2f} | RMSE={metrics['rmse']:.2f} | MAPE={metrics['mape']:.1f}%")

        # ── Prédictions avec bruit ────────────────────────────────────────
        forecast, future_fc = predict_with_noise(model_obj, agg)

        # ── Sauvegarde modèle ─────────────────────────────────────────────
        model_path = os.path.join(MODELS_DIR, f"prophet_gare_{_safe(gare)}.pkl")
        joblib.dump(model_obj, model_path)
        print(f"  Modèle → {model_path}")

        # ── Sauvegarde params.json ────────────────────────────────────────
        params = {
            "name"            : f"prophet_gare_{_safe(gare)}",
            "gare"            : gare,
            "gare_safe"       : _safe(gare),
            "model_type"      : "prophet",
            "model_path"      : model_path,
            "future_hours"    : FUTURE_HOURS,
            "noise_std"       : NOISE_STD,
            "train_ratio"     : TRAIN_RATIO,
            "seasonality_mode": "additive",
            "changepoint_prior_scale": 0.15,
            "fourier_orders"  : {"daily": 10, "weekly": 5, "yearly": 8},
            "data_start"      : str(agg["ds"].min()),
            "data_end"        : str(agg["ds"].max()),
            "n_hours_train"   : n_train,
            "y_mean"          : round(float(agg["y"].mean()), 3),
            "y_std"           : round(float(agg["y"].std()), 3),
            "y_max"           : round(float(agg["y"].max()), 3),
            "metrics"         : metrics,
        }
        params_path = os.path.join(MODELS_DIR, f"prophet_gare_{_safe(gare)}_params.json")
        with open(params_path, "w") as f:
            json.dump(params, f, indent=2)
        print(f"  Params → {params_path}")

        # ── Sauvegarde prévisions ──────────────────────────────────────────
        future_df = future_fc[["ds", "yhat_noisy", "yhat_lower_noisy", "yhat_upper_noisy"]].copy()
        future_df.columns = ["datetime", "predicted", "lower_95", "upper_95"]
        future_df["predicted"] = future_df["predicted"].round(2)
        future_df["lower_95"]  = future_df["lower_95"].round(2)
        future_df["upper_95"]  = future_df["upper_95"].round(2)
        csv_path = os.path.join(MODELS_DIR, f"prophet_gare_{_safe(gare)}_forecast.csv")
        future_df.to_csv(csv_path, index=False)
        print(f"  Forecast CSV → {csv_path}")

        # ── Plot diagnostic ────────────────────────────────────────────────
        plot_diagnostic(gare, agg, forecast, future_fc, metrics)

    # ── Résumé ────────────────────────────────────────────────────────────
    elapsed = int(time.time() - start)
    print(f"\n\n{'='*60}")
    print(f"  RÉSUMÉ — {len(GARES)} modèles Prophet")
    print(f"  Temps total : {elapsed//60}m {elapsed%60}s")
    print(f"{'='*60}")
    print(f"  {'Gare':<20} {'MAE':>8} {'RMSE':>8} {'MAPE':>8}")
    print(f"  {'-'*46}")
    for gare, m in all_metrics.items():
        print(f"  {gare:<20} {m['mae']:>8.2f} {m['rmse']:>8.2f} {m['mape']:>7.1f}%")

    # Index global
    index = {
        "gares"       : GARES,
        "future_hours": FUTURE_HOURS,
        "noise_std"   : NOISE_STD,
        "models"      : {
            g: {
                "model_path" : os.path.join(MODELS_DIR, f"prophet_gare_{_safe(g)}.pkl"),
                "params_path": os.path.join(MODELS_DIR, f"prophet_gare_{_safe(g)}_params.json"),
                "metrics"    : all_metrics[g],
            }
            for g in GARES
        }
    }
    with open(os.path.join(MODELS_DIR, "prophet_gare_index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n  Index → {os.path.join(MODELS_DIR, 'prophet_gare_index.json')}")
    print("\n  Fichiers créés :")
    for fname in sorted(os.listdir(MODELS_DIR)):
        if "prophet_gare" in fname:
            size = os.path.getsize(os.path.join(MODELS_DIR, fname))
            print(f"    {fname:<52} {size//1024:>5} Ko")
    print(f"\n{'='*60}")
    