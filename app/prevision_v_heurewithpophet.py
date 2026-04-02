import pandas as pd
from prophet import Prophet
from pathlib import Path
import plotly.graph_objects as go

# ── Preprocessing ────────────────────────────────────────────────────────
csv_path = Path(__file__).parent / "donnees_synthetiques_tollxpress_benin_2023-2024.csv"
df = pd.read_csv(csv_path)
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
f_passages = model.predict(model.make_future_dataframe(periods=30))


def visualise():
    """Generate interactive forecast plot - Plotly only."""
    fig = go.Figure()
    
    # Add actual data
    fig.add_trace(go.Scatter(
        x=agg["ds"], y=agg["y"],
        mode='lines',
        name='Réel',
        line=dict(color='#0d6efd', width=2)
    ))
    
    # Add forecast
    fig.add_trace(go.Scatter(
        x=f_passages["ds"], y=f_passages["yhat"],
        mode='lines',
        name='Prédit',
        line=dict(color='#dc3545', width=2, dash='dash')
    ))
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=f_passages["ds"],
        y=f_passages["yhat_upper"],
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=f_passages["ds"],
        y=f_passages["yhat_lower"],
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='IC 95%',
        fillcolor='rgba(220, 53, 69, 0.2)'
    ))
    
    fig.update_layout(
        title='Total Passages - Prévisions avec Prophet',
        xaxis_title='Date',
        yaxis_title='Nombre de Passages',
        hovermode='x unified',
        template='plotly_white',
        autosize=True,
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    return fig.to_html(full_html=False)
     


