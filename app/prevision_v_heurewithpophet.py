import numpy as np
import pandas as pd
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app.database import get_engine


FORECAST_HOURS = 168


def _load_hourly_series() -> pd.DataFrame:
    query = """
    SELECT
        date_heure AS "date_heure",
        id_transaction AS "id_transaction"
    FROM fact_transactions
    """
    df = pd.read_sql(query, get_engine())
    df["date_heure"] = pd.to_datetime(df["date_heure"])

    hourly = (
        df.groupby(df["date_heure"].dt.floor("h"))
        .agg(y=("id_transaction", "count"))
        .reset_index()
        .rename(columns={"date_heure": "ds"})
        .sort_values("ds")
    )
    return hourly


def _build_model() -> Prophet:
    model = Prophet(
        seasonality_mode="additive",
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=True,
        interval_width=0.95,
        changepoint_prior_scale=0.06,
    )
    return model


def _compute_quality_metrics(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    if test_df.empty:
        return {"mae": 0.0, "mape": 0.0, "coverage_95": 0.0, "test_points": 0}

    model = _build_model()
    model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test_df), freq="h")
    fc = model.predict(future)
    pred = fc[fc["ds"] > train_df["ds"].max()].copy().head(len(test_df))

    y_true = test_df["y"].to_numpy(dtype=float)
    y_hat = np.clip(pred["yhat"].to_numpy(dtype=float), 0, None)
    y_low = np.clip(pred["yhat_lower"].to_numpy(dtype=float), 0, None)
    y_up = np.clip(pred["yhat_upper"].to_numpy(dtype=float), 0, None)

    mae = float(np.mean(np.abs(y_true - y_hat)))
    non_zero_mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[non_zero_mask] - y_hat[non_zero_mask]) / y_true[non_zero_mask])) * 100.0) if np.any(non_zero_mask) else 0.0
    coverage = float(np.mean((y_true >= y_low) & (y_true <= y_up)) * 100.0)

    return {
        "mae": mae,
        "mape": mape,
        "coverage_95": coverage,
        "test_points": int(len(test_df)),
    }


def _fit_and_forecast(hourly: pd.DataFrame, horizon_hours: int = FORECAST_HOURS) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if len(hourly) < 400:
        raise ValueError("Pas assez de données pour une prévision fiable (minimum 400 points horaires).")

    holdout = min(FORECAST_HOURS, max(24, len(hourly) // 10))
    train_df = hourly.iloc[:-holdout].copy()
    test_df = hourly.iloc[-holdout:].copy()

    quality = _compute_quality_metrics(train_df, test_df)

    full_model = _build_model()
    full_model.fit(hourly)
    future = full_model.make_future_dataframe(periods=horizon_hours, freq="h")
    forecast = full_model.predict(future)

    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(lower=0)

    future_fc = forecast[forecast["ds"] > hourly["ds"].max()].copy()
    return forecast, future_fc, quality


def visualise() -> str:
    hourly = _load_hourly_series()
    forecast, future_fc, quality = _fit_and_forecast(hourly, horizon_hours=FORECAST_HOURS)

    last_hist = hourly[hourly["ds"] >= hourly["ds"].max() - pd.Timedelta(days=14)]

    peak_idx = int(future_fc["yhat"].idxmax())
    peak_row = future_fc.loc[peak_idx]
    forecast_total = float(future_fc["yhat"].sum())

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=last_hist["ds"],
            y=last_hist["y"],
            mode="lines",
            name="Historique (14 jours)",
            line=dict(color="#2563EB", width=1.8),
            opacity=0.75,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=future_fc["ds"],
            y=future_fc["yhat"],
            mode="lines+markers",
            name="Prevision horaire",
            line=dict(color="#DC2626", width=2.5, dash="dash"),
            marker=dict(size=4),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=pd.concat([future_fc["ds"], future_fc["ds"][::-1]]),
            y=pd.concat([future_fc["yhat_upper"], future_fc["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(220,38,38,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Intervalle 95%",
        )
    )

    fig.add_vline(
        x=hourly["ds"].max(),
        line=dict(color="gray", dash="dash", width=1),
    )

    fig.add_annotation(
        x=future_fc["ds"].iloc[min(24, len(future_fc) - 1)],
        y=float(future_fc["yhat"].max()) * 1.04,
        text=(
            f"Total 7j: {forecast_total:,.0f} vehicules<br>"
            f"Pic prevu: {peak_row['ds']} ({peak_row['yhat']:.0f}/h)<br>"
            f"MAE backtest: {quality['mae']:.2f} | MAPE: {quality['mape']:.2f}% | Couverture95: {quality['coverage_95']:.1f}%"
        ),
        showarrow=False,
        align="left",
        bordercolor="#e5e7eb",
        borderwidth=1,
        bgcolor="rgba(255,255,255,0.92)",
    )

    fig.update_layout(
        title="Passages horaires: historique recent + prevision 7 jours",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        margin=dict(t=90, b=50, l=60, r=40),
        legend=dict(orientation="h", y=1.03, x=0),
        xaxis_title="Date",
        yaxis_title="Vehicules/h",
    )

    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    return fig.to_html(full_html=False)
