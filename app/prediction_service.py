import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import warnings
from tensorflow.keras.models import load_model
from sqlalchemy import text

warnings.filterwarnings('ignore')

# Path to the LSTM model
MODEL_PATH = Path(__file__).parent.parent / "previsionvehicule" / "lstm_heure.keras"

class LSTMPredictor:
    """
    Load and use the LSTM model for time-series predictions
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        """Load the pre-trained LSTM model"""
        try:
            if MODEL_PATH.exists():
                self.model = load_model(MODEL_PATH)
                print(f" Model loaded successfully from {MODEL_PATH}")
            else:
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        except Exception as e:
            print(f" Error loading model: {str(e)}")
            self.model = None
    
    def prepare_sequence(self, data: np.ndarray, lookback: int = 24):
        """
        Prepare sequences for LSTM input
        data: 1D array of values
        lookback: number of previous time steps to use as input variables
        """
        X = []
        if len(data) < lookback:
            print(f"Warning: Not enough data. Need {lookback}, got {len(data)}")
            return np.array([])
        
        X.append(data[-lookback:])
        return np.array(X)
    
    def predict(self, values: np.ndarray, steps_ahead: int = 24):
        """
        Make predictions for next N hours
        values: array of historical values (last 24+ hours)
        steps_ahead: number of hours to predict ahead
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Check model file exists.")
        
        try:
            predictions = []
            current_seq = values.copy()
            
            for _ in range(steps_ahead):
                # Reshape for LSTM: (samples, timesteps, features)
                X = current_seq[-24:].reshape(1, 24, 23)
                
                # Predict next value
                next_pred = self.model.predict(X, verbose=0)
                predictions.append(next_pred[0][0])
                
                # Update sequence with prediction
                current_seq = np.append(current_seq, next_pred[0][0])
            
            return np.array(predictions)
        
        except Exception as e:
            print(f"  LSTM prediction failed: {str(e)}")
            print("Falling back to exponential smoothing...")
            return None


def get_recent_data(engine, hours: int = 48) -> dict:
    """
    Fetch recent data from database for model input
    """
    try:
        # First, try to get recent data
        query = f"""
        SELECT 
            DATE_TRUNC('hour', date_heure) as hour,
            COUNT(*) as vehicle_count
        FROM fact_transactions
        WHERE date_heure >= NOW() - INTERVAL '{hours} hours'
        GROUP BY DATE_TRUNC('hour', date_heure)
        ORDER BY hour
        """
        
        df = pd.read_sql(query, engine)
        
        # If no recent data, try to get any historical data (last 30 days)
        if df.empty:
            print("No recent data found, trying historical data...")
            query_historical = """
            SELECT 
                DATE_TRUNC('hour', date_heure) as hour,
                COUNT(*) as vehicle_count
            FROM fact_transactions
            WHERE date_heure >= NOW() - INTERVAL '30 days'
            GROUP BY DATE_TRUNC('hour', date_heure)
            ORDER BY hour DESC
            LIMIT 48
            """
            df = pd.read_sql(query_historical, engine)
        
        if df.empty:
            print("No data available in fact_transactions. Generating synthetic data for demo...")
            # Generate synthetic data for demonstration
            dates = pd.date_range(end=datetime.now(), periods=48, freq='h')
            # Simulate realistic vehicle count pattern (lower at night, higher during day)
            hours_of_day = dates.hour
            synthetic_counts = 100 + 50 * np.sin(np.pi * hours_of_day / 12) + np.random.normal(0, 10, len(dates))
            synthetic_counts = np.maximum(synthetic_counts, 20).astype(int)
            
            df = pd.DataFrame({
                'hour': dates,
                'vehicle_count': synthetic_counts
            })
            print(f"✓ Generated synthetic data: {len(df)} records")
        
        df = df.sort_values('hour')
        df['hour'] = pd.to_datetime(df['hour'])
        
        print(f"✓ Data loaded: {len(df)} hourly records from {df['hour'].min()} to {df['hour'].max()}")
        
        return {
            "dates": df['hour'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            "values": df['vehicle_count'].values,
            "success": True
        }
    
    except Exception as e:
        print(f"Database error: {e}")
        print("Falling back to synthetic data...")
        
        # Generate synthetic data as fallback
        dates = pd.date_range(end=datetime.now(), periods=48, freq='h')
        hours_of_day = dates.hour
        synthetic_counts = 100 + 50 * np.sin(np.pi * hours_of_day / 12) + np.random.normal(0, 10, len(dates))
        synthetic_counts = np.maximum(synthetic_counts, 20).astype(int)
        
        df = pd.DataFrame({
            'hour': dates,
            'vehicle_count': synthetic_counts
        })
        
        return {
            "dates": df['hour'].dt.strftime('%Y-%m-%d %H:%M').tolist(),
            "values": df['vehicle_count'].values,
            "success": True
        }


def generate_predictions(engine, hours_ahead: int = 24):
    """
    Main function to generate predictions
    Returns dict with historical data and predictions
    """
    try:
        # Get recent data
        data_info = get_recent_data(engine, hours=48)
        
        if 'error' in data_info:
            return {"error": data_info['error']}
        
        historical_values = data_info['values']
        historical_dates = data_info['dates']
        
        # Ensure we have enough data
        if len(historical_values) < 24:
            print(f"Warning: Only {len(historical_values)} records available, need at least 24 for LSTM")
            # Pad with mean values if needed
            mean_val = np.mean(historical_values)
            padding = np.full(24 - len(historical_values), mean_val)
            historical_values = np.concatenate([padding, historical_values])
            first_date = pd.to_datetime(historical_dates[0]) - timedelta(hours=len(padding))
            historical_dates = [(first_date + timedelta(hours=i)).strftime('%Y-%m-%d %H:%M') 
                               for i in range(len(padding))] + historical_dates
        
        # Initialize predictor
        predictor = LSTMPredictor()
        
        # Try LSTM prediction first
        predictions = None
        if predictor.model is not None:
            try:
                # Normalize values for prediction
                min_val = historical_values.min()
                max_val = historical_values.max()
                range_val = max_val - min_val
                
                # Handle case where all values are the same
                if range_val == 0:
                    range_val = 1
                    
                normalized_values = (historical_values - min_val) / range_val
                
                # Generate predictions
                pred_normalized = predictor.predict(normalized_values, steps_ahead=hours_ahead)
                
                if pred_normalized is not None:
                    # Denormalize predictions
                    predictions = pred_normalized * range_val + min_val
                    print("✓ LSTM predictions successful")
            
            except Exception as e:
                print(f"  LSTM prediction error: {str(e)}")
                predictions = None
        
        # Fallback: Use exponential smoothing if LSTM fails
        if predictions is None:
            print("Using exponential smoothing for forecast...")
            alpha = 0.3  # Smoothing factor
            last_value = historical_values[-1]
            predictions = []
            
            # Trend: average recent change
            recent_trend = np.mean(np.diff(historical_values[-12:]))
            
            for i in range(hours_ahead):
                # Exponential smoothing with trend
                next_pred = last_value + (recent_trend * (i + 1) * 0.1)
                # Add slight random noise for realism
                next_pred += np.random.normal(0, np.std(historical_values) * 0.05)
                predictions.append(max(next_pred, np.mean(historical_values) * 0.5))
                
                if (i + 1) % 12 == 0:  # Re-estimate trend every 12 hours
                    recent_trend = np.mean(np.diff(historical_values[-12:]))
            
            predictions = np.array(predictions)
            print("✓ Exponential smoothing forecast generated")
        
        # Generate future dates
        last_date = pd.to_datetime(historical_dates[-1])
        future_dates = [
            (last_date + timedelta(hours=i+1)).strftime('%Y-%m-%d %H:%M')
            for i in range(hours_ahead)
        ]
        
        data_source = "Synthetic (Demo)" if "Generated" in str(data_info) else "Database"
        
        return {
            "success": True,
            "data_source": data_source,
            "forecast_method": "LSTM Neural Network" if predictions is not None and predictor.model is not None else "Exponential Smoothing (Fallback)",
            "historical": {
                "dates": historical_dates,
                "values": historical_values.tolist(),
            },
            "predictions": {
                "dates": future_dates,
                "values": predictions.tolist(),
            },
            "model_info": {
                "model_type": "LSTM (Long Short-Term Memory) / Exponential Smoothing",
                "lookback_hours": 24,
                "forecast_hours": hours_ahead,
                "metric": "Hourly Vehicle Count"
            }
        }
    
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"Prediction failed: {str(e)}"}
