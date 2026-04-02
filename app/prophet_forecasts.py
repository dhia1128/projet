import pandas as pd
import numpy as np
from prophet import Prophet
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Chargement unique des données ─────────────────────────────────────────
CSV_PATH = Path(__file__).parent / "donnees_synthetiques_tollxpress_benin_2023-2024.csv"

def _load() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df["Date_Heure"] = pd.to_datetime(df["Date_Heure"])
    return df


def _fit_predict(agg: pd.DataFrame, periods: int = 30, freq: str = "D") -> tuple:
    """
    Entraîne Prophet sur un DataFrame {ds, y} et prédit `periods` périodes.
    Retourne (model, forecast, future_fc) avec yhat clippé à 0.
    """
    model = Prophet(
        seasonality_mode="additive",
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=(freq == "h"),
        interval_width=0.95,
        changepoint_prior_scale=0.05,
    )
    model.fit(agg)
    future   = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(lower=0)
    future_fc = forecast[forecast["ds"] > agg["ds"].max()].copy()
    return model, forecast, future_fc


# ══════════════════════════════════════════════════════════════════════════
# FONCTION 1 — Répartition par classe de véhicule
# ══════════════════════════════════════════════════════════════════════════
def plot_classe() -> str:
    """
    Un subplot par classe (MOTO, VL, PL, BUS).
    Granularité journalière, prévision 30 jours.
    Retourne HTML Plotly.
    """
    df = _load()
    classes = ["MOTO", "VL", "PL", "BUS"]
    colors  = {"MOTO": "#2563EB", "VL": "#059669", "PL": "#D97706", "BUS": "#DC2626"}

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"Classe — {c}" for c in classes],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    positions = {0: (1,1), 1: (1,2), 2: (2,1), 3: (2,2)}

    for i, classe in enumerate(classes):
        row, col = positions[i]
        color = colors[classe]

        # agrégation journalière pour cette classe
        agg = (
            df[df["Classe_vehicule"] == classe]
              .groupby(df["Date_Heure"].dt.normalize())
              .agg(y=("ID_transaction", "count"))
              .reset_index()
              .rename(columns={"Date_Heure": "ds"})
        )

        _, _, future_fc = _fit_predict(agg, periods=30, freq="D")

        # historique (dernier mois)
        last_month = agg[agg["ds"] >= agg["ds"].max() - pd.Timedelta(days=60)]
        fig.add_trace(go.Scatter(
            x=last_month["ds"], y=last_month["y"],
            mode="lines", name=f"{classe} — Réel",
            line=dict(color=color, width=1.5), opacity=0.6,
            legendgroup=classe, showlegend=(i == 0),
        ), row=row, col=col)

        # prévision
        fig.add_trace(go.Scatter(
            x=future_fc["ds"], y=future_fc["yhat"],
            mode="lines+markers", name=f"{classe} — Prévision",
            line=dict(color=color, width=2.5, dash="dash"),
            marker=dict(size=5),
            legendgroup=classe, showlegend=False,
        ), row=row, col=col)

        # IC 95%
        fig.add_trace(go.Scatter(
            x=pd.concat([future_fc["ds"], future_fc["ds"][::-1]]),
            y=pd.concat([future_fc["yhat_upper"], future_fc["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor=f"rgba{tuple(list(_hex_to_rgb(color)) + [0.15])}",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, legendgroup=classe,
        ), row=row, col=col)

        # ligne de séparation historique / futur
        fig.add_vline(x=agg["ds"].max(),
                      line=dict(color="gray", dash="dash", width=1),
                      row=row, col=col)

        # annotation pic
        peak = future_fc.loc[future_fc["yhat"].idxmax(), "ds"]
        peak_val = future_fc["yhat"].max()
        fig.add_annotation(
            x=peak, y=peak_val,
            text=f"Pic: {peak_val:.0f}",
            showarrow=True, arrowhead=2, arrowsize=0.8,
            font=dict(size=10, color=color),
            row=row, col=col,
        )

    fig.update_layout(
        title=dict(
            text="Prophet — Prévision 30 jours par classe de véhicule",
            font=dict(size=15),
        ),
        height=720,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=90, b=50, l=60, r=40),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", tickangle=30)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig.to_html(full_html=False)


# ══════════════════════════════════════════════════════════════════════════
# FONCTION 2 — Trafic par gare
# ══════════════════════════════════════════════════════════════════════════
def plot_gare() -> str:
    """
    Un subplot par gare (6 gares).
    Granularité journalière, prévision 30 jours.
    Retourne HTML Plotly.
    """
    df    = _load()
    gares  = ["Cotonou-Nord", "Allada", "Houegbo", "Porto-Novo", "Epke", "Parakou"]
    colors = ["#2563EB", "#059669", "#D97706", "#DC2626", "#7C3AED", "#0891B2"]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=gares,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )
    positions = {0:(1,1), 1:(1,2), 2:(2,1), 3:(2,2), 4:(3,1), 5:(3,2)}

    for i, (gare, color) in enumerate(zip(gares, colors)):
        row, col = positions[i]

        agg = (
            df[df["Gare"] == gare]
              .groupby(df["Date_Heure"].dt.normalize())
              .agg(y=("ID_transaction", "count"))
              .reset_index()
              .rename(columns={"Date_Heure": "ds"})
        )

        _, _, future_fc = _fit_predict(agg, periods=30, freq="D")

        # historique (60 derniers jours)
        last_hist = agg[agg["ds"] >= agg["ds"].max() - pd.Timedelta(days=60)]
        fig.add_trace(go.Scatter(
            x=last_hist["ds"], y=last_hist["y"],
            mode="lines", name=f"{gare} — Réel",
            line=dict(color=color, width=1.5), opacity=0.65,
            showlegend=False,
        ), row=row, col=col)

        # prévision
        fig.add_trace(go.Scatter(
            x=future_fc["ds"], y=future_fc["yhat"],
            mode="lines+markers", name=f"{gare} — Prévision",
            line=dict(color=color, width=2.5, dash="dash"),
            marker=dict(size=5),
            showlegend=False,
        ), row=row, col=col)

        # IC 95%
        fig.add_trace(go.Scatter(
            x=pd.concat([future_fc["ds"], future_fc["ds"][::-1]]),
            y=pd.concat([future_fc["yhat_upper"], future_fc["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor=f"rgba{tuple(list(_hex_to_rgb(color)) + [0.15])}",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
        ), row=row, col=col)

        fig.add_vline(x=agg["ds"].max(),
                      line=dict(color="gray", dash="dash", width=1),
                      row=row, col=col)

        # annotation total prévu
        total = future_fc["yhat"].sum()
        fig.add_annotation(
            x=future_fc["ds"].iloc[15], y=future_fc["yhat"].max(),
            text=f"Total 30j: {total:.0f}",
            showarrow=False, font=dict(size=10, color=color),
            row=row, col=col,
        )

    fig.update_layout(
        title=dict(
            text="Prophet — Prévision 30 jours par gare",
            font=dict(size=15),
        ),
        height=950,
        template="plotly_white",
        hovermode="x unified",
        margin=dict(t=90, b=50, l=60, r=40),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", tickangle=30)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")
    return fig.to_html(full_html=False)


# ══════════════════════════════════════════════════════════════════════════
# FONCTION 3 — Trafic global réseau
# ══════════════════════════════════════════════════════════════════════════
def plot_global() -> str:
    """
    Vue globale du réseau avec 3 subplots :
      - Forecast global 30 jours
      - Comparaison des gares (barres empilées sur les 30 jours prévus)
      - Profil hebdomadaire moyen prévu
    Retourne HTML Plotly.
    """
    df = _load()

    # ── agrégation globale journalière ────────────────────────────────────
    agg_global = (
        df.groupby(df["Date_Heure"].dt.normalize())
          .agg(y=("ID_transaction", "count"))
          .reset_index()
          .rename(columns={"Date_Heure": "ds"})
    )

    _, _, future_global = _fit_predict(agg_global, periods=30, freq="D")

    # ── par gare (pour barres empilées) ───────────────────────────────────
    gares  = ["Cotonou-Nord", "Allada", "Houegbo", "Porto-Novo", "Epke", "Parakou"]
    colors = ["#2563EB", "#059669", "#D97706", "#DC2626", "#7C3AED", "#0891B2"]
    gare_forecasts = {}
    for gare in gares:
        agg_g = (
            df[df["Gare"] == gare]
              .groupby(df["Date_Heure"].dt.normalize())
              .agg(y=("ID_transaction", "count"))
              .reset_index()
              .rename(columns={"Date_Heure": "ds"})
        )
        _, _, fc = _fit_predict(agg_g, periods=30, freq="D")
        gare_forecasts[gare] = fc

    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            "Trafic global réseau — Historique + Prévision 30 jours",
            "Prévision par gare — 30 prochains jours (barres empilées)",
            "Profil hebdomadaire moyen prévu (réseau total)",
        ),
        vertical_spacing=0.12,
        row_heights=[0.4, 0.35, 0.25],
    )

    # ── (1) Global forecast ───────────────────────────────────────────────
    last_hist = agg_global[agg_global["ds"] >= agg_global["ds"].max() - pd.Timedelta(days=90)]
    fig.add_trace(go.Scatter(
        x=last_hist["ds"], y=last_hist["y"],
        mode="lines", name="Réel (3 derniers mois)",
        line=dict(color="#2563EB", width=1.8), opacity=0.7,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=future_global["ds"], y=future_global["yhat"],
        mode="lines+markers", name="Prévision réseau 30j",
        line=dict(color="#DC2626", width=2.5, dash="dash"),
        marker=dict(size=6),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=pd.concat([future_global["ds"], future_global["ds"][::-1]]),
        y=pd.concat([future_global["yhat_upper"], future_global["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(220,38,38,0.12)",
        line=dict(color="rgba(0,0,0,0)"), name="IC 95%",
    ), row=1, col=1)

    fig.add_vline(x=agg_global["ds"].max(),
                  line=dict(color="gray", dash="dash", width=1),
                  row=1, col=1)

    # annotation total réseau
    total_reseau = future_global["yhat"].sum()
    fig.add_annotation(
        x=future_global["ds"].iloc[10],
        y=future_global["yhat"].max() * 1.05,
        text=f"Total réseau prévu : {total_reseau:,.0f} véhicules",
        showarrow=False, font=dict(size=12, color="#DC2626"),
        bgcolor="rgba(255,255,255,0.8)",
        row=1, col=1,
    )

    # ── (2) Barres empilées par gare ──────────────────────────────────────
    for gare, color in zip(gares, colors):
        fc = gare_forecasts[gare]
        fig.add_trace(go.Bar(
            x=fc["ds"], y=fc["yhat"],
            name=gare,
            marker_color=color,
            opacity=0.85,
        ), row=2, col=1)

    fig.update_layout(barmode="stack")

    # ── (3) Profil hebdomadaire moyen ─────────────────────────────────────
    future_global["weekday"]     = future_global["ds"].dt.day_name()
    future_global["weekday_num"] = future_global["ds"].dt.dayofweek

    weekly_avg = (
        future_global.groupby(["weekday_num", "weekday"])["yhat"]
                     .mean()
                     .reset_index()
                     .sort_values("weekday_num")
    )
    day_labels = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
    weekly_avg["weekday_fr"] = day_labels

    bar_colors = [
        "#DC2626" if d in ["Samedi","Sunday","Dimanche","Saturday"]
        else "#2563EB"
        for d in weekly_avg["weekday"]
    ]

    fig.add_trace(go.Bar(
        x=weekly_avg["weekday_fr"], y=weekly_avg["yhat"],
        name="Moy/jour semaine",
        marker_color=bar_colors,
        showlegend=False,
    ), row=3, col=1)

    fig.add_hline(
        y=weekly_avg["yhat"].mean(),
        line=dict(color="gray", dash="dash", width=1),
        row=3, col=1,
    )
    fig.add_annotation(
        x=6, y=weekly_avg["yhat"].mean(),
        text=f"Moy: {weekly_avg['yhat'].mean():.0f}",
        showarrow=False, font=dict(size=10, color="gray"),
        row=3, col=1,
    )

    fig.update_layout(
        title=dict(
            text="Prophet — Trafic global réseau TollXpress · Prévision 30 jours",
            font=dict(size=15),
        ),
        height=980,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.03, x=0),
        margin=dict(t=90, b=50, l=60, r=40),
        barmode="stack",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", tickangle=30)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    return fig.to_html(full_html=False)


# ── Helper couleur hex → rgb ───────────────────────────────────────────────
def _hex_to_rgb(hex_color: str) -> tuple:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


# ════════════════════════════════════════════════════════════════════════════
# INTEGRATION FLASK — exemple d'utilisation
# ════════════════════════════════════════════════════════════════════════════
"""
from flask import Flask
from prophet_forecasts import plot_classe, plot_gare, plot_global

app = Flask(__name__)

@app.route("/plot/classe")   
def route_classe():  return plot_classe()

@app.route("/plot/gare")     
def route_gare():    return plot_gare()

@app.route("/plot/global")   
def route_global():  return plot_global()
"""

# ── Lancement direct — ouvre les 3 plots dans le navigateur ───────────────
if __name__ == "__main__":
    import webbrowser, tempfile, os

    PLOTLY_JS = "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>"

    for name, fn in [("classe", plot_classe), ("gare", plot_gare), ("global", plot_global)]:
        print(f"Génération : {name} ...")
        html = f"<html><head>{PLOTLY_JS}</head><body>{fn()}</body></html>"
        tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{name}.html", mode="w", encoding="utf-8")
        tmp.write(html)
        tmp.close()
        webbrowser.open(f"file://{tmp.name}")
        print(f"  → {tmp.name}")
