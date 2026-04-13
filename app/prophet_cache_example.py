"""
Example: Applying joblib caching to Prophet predictions
Shows how to avoid retraining Prophet models
"""

from prophet import Prophet
import pandas as pd
from app.model_cache import save_model, load_model
from app.database import get_engine

MODEL_NAME = "prophet_trafic_global"


def predict_with_prophet_cached(days: int = 30) -> pd.DataFrame:
    """
    Prophet forecast with joblib caching
    
    First call (5-10 sec):   Train Prophet + save cache
    Later calls (< 1 sec):   Load cache + predict instantly
    """
    
    # 1. Try loading cached Prophet model
    cached_prophet, metadata = load_model(MODEL_NAME)
    
    if cached_prophet is not None:
        # ⚡ Model found in cache - instant load!
        print(f"⚡ Loaded Prophet from cache (instant)")
        model = cached_prophet
    else:
        # First time: train Prophet
        print(f"🔄 Training Prophet model (takes 5-10 sec)...")
        
        # Load data from database
        df = pd.read_sql(
            "SELECT DATE(Date_Heure) AS ds, COUNT(ID_transaction) AS y FROM fact_transactions GROUP BY DATE(Date_Heure)",
            get_engine()
        )
        df["ds"] = pd.to_datetime(df["ds"])
        
        # Train Prophet
        model = Prophet(interval_width=0.95, yearly_seasonality=True)
        model.fit(df)
        
        # Save model cache for next time
        save_model(model, MODEL_NAME)
        print(f"✅ Model cached to models/{MODEL_NAME}.joblib")
    
    # 2. Make forecast
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


# Usage in web API:
# @app.get("/forecast/prophet")
# def get_prophet_forecast():
#     forecast = predict_with_prophet_cached(days=30)
#     return forecast.to_json()
#
# Performance:
# - 1st request: ~8 seconds (training)
# - All later requests: ~200ms (cached + predict only)
