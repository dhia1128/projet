import pandas as pd
import numpy as np
from prophet import Prophet
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
from app.database import get_engine


GARE_COLORS = {
    "Cotonou-Nord": "#2563EB",
    "Allada"      : "#059669",
    "Houegbo"     : "#D97706",
    "Porto-Novo"  : "#DC2626",
    "Epke"        : "#7C3AED",
    "Parakou"     : "#0891B2",
}




def _load_daily_global() -> pd.DataFrame:
    """Retourne un DataFrame (ds, y) agrégé par jour sur tout le réseau."""
    df = pd.read_sql("SELECT DATE(Date_Heure) AS ds, COUNT(ID_transaction) AS y FROM fact_transactions GROUP BY DATE(Date_Heure)", get_engine())
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def _fit_prophet(agg: pd.DataFrame, days: int = 30) -> tuple:
    """Entraîne Prophet et retourne (model, hist_fc, future_fc)."""
    model = Prophet(
        seasonality_mode="additive",
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.95,
        changepoint_prior_scale=0.05,
    )
    model.fit(agg)
    future   = model.make_future_dataframe(periods=days, freq="D")
    forecast = model.predict(future)
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(lower=0)
    hist_fc   = forecast[forecast["ds"] <= agg["ds"].max()].copy()
    future_fc = forecast[forecast["ds"] >  agg["ds"].max()].copy()
    return model, hist_fc, future_fc


# ════════════════════════════════════════════════════════════════════════════

def predict_global_reseau(days: int = 30) -> str:
    """
    Prévision du trafic total réseau sur 30 jours.
    Retourne HTML Plotly avec 2 vues :
      - (1,1) Prévision journalière — barres journalières prévues avec IC
      - (1,2) Profil hebdomadaire moyen prévu (lun → dim)
    """
    agg = _load_daily_global()
    model, hist_fc, future_fc = _fit_prophet(agg, days=days)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Prévision journalière — {days} prochains jours",
            "Profil hebdomadaire moyen prévu",
        ),
        vertical_spacing=0.16,
        horizontal_spacing=0.12,
    )

    # ── (1,1) Barres journalières ─────────────────────────────────────────
    bar_colors = [
        "#DC2626" if d.weekday() >= 5 else "#2563EB"
        for d in future_fc["ds"]
    ]
    fig.add_trace(go.Bar(
        x=future_fc["ds"], y=future_fc["yhat"],
        name="Véhicules/jour",
        marker_color=bar_colors,
        opacity=0.85,
        error_y=dict(
            type="data", symmetric=False,
            array=(future_fc["yhat_upper"] - future_fc["yhat"]).values,
            arrayminus=(future_fc["yhat"] - future_fc["yhat_lower"]).values,
            color="rgba(0,0,0,0.25)",
        ),
        showlegend=False,
    ), row=1, col=1)

    # légende couleur weekend
    for label, color in [("Semaine", "#2563EB"), ("Weekend", "#DC2626")]:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(color=color, size=10, symbol="square"),
            name=label, showlegend=True,
        ), row=1, col=1)

    # ── (1,2) Profil hebdomadaire ─────────────────────────────────────────
    future_fc["weekday_num"] = future_fc["ds"].dt.dayofweek
    weekly = (
        future_fc.groupby("weekday_num")[["yhat", "yhat_lower", "yhat_upper"]]
                 .mean()
                 .reindex(range(7), fill_value=0)
    )
    jours_fr = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
    wk_colors = ["#DC2626" if j >= 5 else "#2563EB" for j in range(7)]

    fig.add_trace(go.Bar(
        x=jours_fr, y=weekly["yhat"].values,
        marker_color=wk_colors, opacity=0.85,
        name="Moy/jour semaine",
        error_y=dict(
            type="data", symmetric=False,
            array=(weekly["yhat_upper"] - weekly["yhat"]).values,
            arrayminus=(weekly["yhat"] - weekly["yhat_lower"]).values,
            color="rgba(0,0,0,0.25)",
        ),
        showlegend=False,
    ), row=1, col=2)

    # ligne moyenne globale
    moy_global = weekly["yhat"].mean()
    fig.add_hline(
        y=moy_global,
        line=dict(color="#94a3b8", dash="dash", width=1.2),
        row=1, col=2,
    )
    fig.add_annotation(
        x=6, y=moy_global * 1.04,
        text=f"Moy: {moy_global:.0f}",
        showarrow=False, font=dict(size=10, color="#64748b"),
        row=1, col=2,
    )

    fig.update_layout(
        title=dict(
            text=f"Prophet — Trafic Global Réseau TollXpress · Prévision {days} jours",
            font=dict(size=16),
        ),
        height=820,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0),
        margin=dict(t=95, b=50, l=60, r=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", tickangle=30)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")

    return fig.to_html(full_html=False)



