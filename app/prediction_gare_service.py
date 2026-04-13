import pandas as pd
import numpy as np
from prophet import Prophet
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
from app.database import get_engine
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from app.model_cache import save_model, load_model, model_exists

# Reuse your existing data loading function
def _load_daily_global() -> pd.DataFrame:
    """Same as before"""
    df = pd.read_sql("SELECT DATE(Date_Heure) AS ds, COUNT(ID_transaction) AS y FROM fact_transactions GROUP BY DATE(Date_Heure)", get_engine())
    df["ds"] = pd.to_datetime(df["ds"])
    return df


def _prepare_lstm_data(data: pd.Series, time_step: int = 7):
    """Create sequences for LSTM"""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled_data) - time_step):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)   # (samples, time_steps, features)

    return X, y, scaler


def _build_lstm_model(time_step: int = 7):
    """Build LSTM model - simplified for small datasets"""
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.1))
    model.add(LSTM(16, return_sequences=False))
    model.add(Dropout(0.1))
    model.add(Dense(8))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def predict_global_reseau_lstm(days: int = 30, time_step: int = 7) -> str:
    """
    Prévision du trafic total avec LSTM (Deep Learning)
    Loads cached model on subsequent calls (⚡ ~10-100x faster)
    Retourne un graphique Plotly similaire à Prophet
    """
    # 1. Load data
    agg = _load_daily_global()
    agg = agg.sort_values('ds').reset_index(drop=True)
    
    MODEL_NAME = f"lstm_reseau_global_ts{time_step}"
    
    # 2. Try to load cached model
    cached_model, metadata = load_model(MODEL_NAME)
    
    if cached_model is not None:
        # ⚡ Model found in cache - use it directly!
        model = cached_model
        scaler = metadata.get("scaler")
        print(f"⚡ Using cached model (instant load)")
    else:
        # First time: prepare data and train
        print(f"🔄 Training new LSTM model...")
        X, y, scaler = _prepare_lstm_data(agg['y'], time_step=time_step)
        
        # Split into train/test (80% train)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build and train model
        model = _build_lstm_model(time_step)
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=4,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Save model + scaler to cache (⚡ for next time)
        save_model(model, MODEL_NAME, scaler=scaler)
    
    # 5. Forecast next 'days' days with noise decay to prevent convergence
    last_sequence = agg['y'].values[-time_step:].reshape(-1, 1)
    last_sequence_scaled = scaler.transform(last_sequence)
    
    # Calculate trend from recent data for better predictions
    recent_trend = (agg['y'].values[-1] - agg['y'].values[-7]) / 7 if len(agg) >= 7 else 0
    
    future_predictions = []
    current_seq = last_sequence_scaled.copy()
    
    for step in range(days):
        pred = model.predict(current_seq.reshape(1, time_step, 1), verbose=0)
        future_predictions.append(pred[0, 0])
        
        # Add decaying noise to prevent autoregressive collapse
        noise = np.random.normal(0, 0.01 * (1 - step / days))
        pred_with_noise = pred + noise
        
        # Update sequence (shift + add new prediction)
        current_seq = np.append(current_seq[1:], pred_with_noise).reshape(-1, 1)
    
    # Inverse transform predictions
    future_pred = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
    future_pred = np.maximum(future_pred, 0)   # No negative transactions
    
    # Apply slight trend to add variation
    trend_boost = np.linspace(0, recent_trend * days * 0.1, days)
    future_pred = future_pred + trend_boost
    
    # Create future dates
    last_date = agg['ds'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days, freq='D')
    
    # 6. Create Plotly figure (similar to your Prophet version)
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Prévision LSTM — {days} prochains jours",
            "Profil hebdomadaire moyen prévu (LSTM)"
        ),
        horizontal_spacing=0.12,
    )

    # Left plot: Daily forecast
    bar_colors = ["#DC2626" if d.weekday() >= 5 else "#2563EB" for d in future_dates]
    
    fig.add_trace(go.Bar(
        x=future_dates, 
        y=future_pred,
        name="Véhicules/jour (LSTM)",
        marker_color=bar_colors,
        opacity=0.85,
    ), row=1, col=1)

    # Right plot: Weekly profile
    future_df = pd.DataFrame({'ds': future_dates, 'yhat': future_pred})
    future_df['weekday_num'] = future_df['ds'].dt.dayofweek
    
    weekly = future_df.groupby('weekday_num')['yhat'].mean().reindex(range(7), fill_value=0)
    jours_fr = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
    wk_colors = ["#DC2626" if j >= 5 else "#2563EB" for j in range(7)]

    fig.add_trace(go.Bar(
        x=jours_fr, 
        y=weekly.values,
        marker_color=wk_colors,
        opacity=0.85,
        name="Moyenne par jour de semaine"
    ), row=1, col=2)

    # Global average line
    moy_global = weekly.mean()
    fig.add_hline(y=moy_global, line=dict(color="#94a3b8", dash="dash"), row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"LSTM — Prévision Trafic Global Réseau TollXpress ({days} jours)",
            font=dict(size=16)
        ),
        height=820,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0),
    )

    return fig.to_html(full_html=False)