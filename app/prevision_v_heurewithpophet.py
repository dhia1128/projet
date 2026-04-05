import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from app.database import get_engine

# ── Preprocessing ────────────────────────────────────────────────────────
query = """
SELECT date_heure AS "Date_Heure", id_transaction AS "ID_transaction"
FROM fact_transactions
"""
df = pd.read_sql(query, get_engine())
df["Date_Heure"] = pd.to_datetime(df["Date_Heure"])

agg = (
    df.groupby(df["Date_Heure"].dt.floor("h"))
      .agg(y=("ID_transaction", "count"))
      .reset_index()
      .rename(columns={"Date_Heure": "ds"})
)

# ── Modèle passages ──────────────────────────────────────────────────────
model = Prophet(seasonality_mode="multiplicative",
                     weekly_seasonality=True,
                     yearly_seasonality=True,
                     interval_width=0.95)
model.fit(agg)
f_passages = model.predict(model.make_future_dataframe(periods=168))  # 168 hours = 7 days

future    = model.make_future_dataframe(periods=168, freq="h")  # 168 hours = 7 days
forecast  = model.predict(future)
 
# séparer historique et future
hist_fc   = forecast[forecast["ds"] <= agg["ds"].max()]
future_fc = forecast[forecast["ds"] >  agg["ds"].max()]

 
def visualise():
    """Plot simple forecast uniquement (pour Flask route /plot/forecast)"""
    fig = go.Figure()
 
    fig.add_trace(go.Scatter(
        x=agg["ds"], y=agg["y"],
        mode="lines", name="Réel",
        line=dict(color="#2563EB", width=1.2), opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=future_fc["ds"], y=future_fc["yhat"],
        mode="lines+markers", name="Prévision 30 jours",
        line=dict(color="#DC2626", width=2.5),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([future_fc["ds"], future_fc["ds"][::-1]]),
        y=pd.concat([future_fc["yhat_upper"], future_fc["yhat_lower"][::-1]]),
        fill="toself", fillcolor="rgba(220,38,38,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="IC 95%",
    ))
    fig.add_vline(x=agg["ds"].max(),
                  line=dict(color="gray", dash="dash", width=1))
    fig.add_annotation(
        x=agg["ds"].max(), y=agg["y"].max(),
        text="  Début prévision", showarrow=False,
        font=dict(color="gray", size=11),
    )
    fig.update_layout(
        title="Prévision Prophet — 7 prochains jours (horaire)",
        xaxis_title="Date", yaxis_title="Nb véhicules/heure",
        xaxis_rangeslider_visible=True,
        hovermode="x unified", template="plotly_white",
        height=600, margin=dict(t=80, b=50),
    )
    return fig.to_html(full_html=False)
    
     


