import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from app.database import engine
import io
import base64
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="LSTM Traffic Prediction API", version="1.0.0")

# =============================================================================
# LOAD SAVED MODEL AND GENERATE PREDICTIONS
# =============================================================================

def load_and_predict(
    model_path='lstm_keras_best.keras',
    lookback=23,
    future_hours=7*24,
    noise_std=0.05
):
    """
    Load a saved LSTM model and generate predictions.
    
    Args:
        model_path (str): Path to the saved model
        lookback (int): Number of past hours to use for prediction (23 by default)
        future_hours (int): Number of future hours to predict (default: 7 days)
        noise_std (float): Standard deviation for Gaussian noise
    
    Returns:
        dict: Dictionary containing predictions, dates, and metadata
    """
    
    print("\n" + "="*70)
    print("LSTM MODEL PREDICTION PIPELINE")
    print("="*70)
    
    try:
        # Step 1: Load the model
        print("\n1️⃣  Loading model...")
        abs_model_path = os.path.join(os.path.dirname(__file__), model_path)
        if not os.path.exists(abs_model_path):
            raise FileNotFoundError(f"Model not found at {abs_model_path}")
        
        model = load_model(abs_model_path)
        print(f"✅ Model loaded from: {abs_model_path}")
        print(f"   Model summary:")
        model.summary()
        
        # Step 2: Load and prepare data
        print(f"\n2️⃣  Loading data from database...")
        df = pd.read_sql_query("SELECT * FROM fact_transactions LIMIT 1", engine)
        print(f"   Available columns: {list(df.columns)}")
        
        # Find date column
        date_col = None
        for col in ['Date_Heure', 'date_heure', 'datetime', 'timestamp']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col is None:
            raise ValueError(f"No date column found. Available: {list(df.columns)}")
        
        # Load all transactions
        df = pd.read_sql_query(f"SELECT {date_col} FROM fact_transactions ORDER BY {date_col}", engine)
        df[date_col] = pd.to_datetime(df[date_col])
        df.set_index(date_col, inplace=True)
        
        # Aggregate by hour
        hourly = df.resample('h').size()
        hourly = pd.DataFrame({'nombre_voitures': hourly})
        hourly = hourly[hourly['nombre_voitures'] > 0]
        
        print(f"✅ Data loaded: {len(hourly)} hours from {hourly.index[0]} to {hourly.index[-1]}")
        
        # Step 3: Fit scaler
        print(f"\n3️⃣  Fitting scaler...")
        scaler = MinMaxScaler()
        scaler.fit(hourly[['nombre_voitures']])
        print(f"✅ Scaler fitted")
        
        # Adjust lookback if needed
        if len(hourly) < lookback:
            print(f"⚠️  Warning: Only {len(hourly)} hours available, need {lookback}")
            lookback = len(hourly)
        
        # Step 4: Generate predictions
        print(f"\n4️⃣  Generating predictions...")
        print(f"   Lookback: {lookback} hours")
        print(f"   Future: {future_hours} hours")
        print(f"   Noise std: {noise_std}")
        
        last_known_scaled_data = scaler.transform(hourly['nombre_voitures'].values[-lookback:].reshape(-1, 1))
        input_seq = last_known_scaled_data.copy()
        
        future_predictions_scaled = []
        future_predictions_clean = []
        
        for step in range(future_hours):
            x_input = input_seq.reshape(1, lookback, 1)
            pred_scaled = model.predict(x_input, verbose=0)[0, 0]
            future_predictions_clean.append(pred_scaled)
            
            # Add noise for dynamic predictions
            noise = np.random.normal(0, noise_std)
            pred_noisy = np.clip(pred_scaled + noise, 0.0, 1.0)
            future_predictions_scaled.append(pred_noisy)
            
            # Update input sequence
            input_seq = np.append(input_seq[1:], [[pred_noisy]], axis=0)
        
        # Denormalize
        predictions = scaler.inverse_transform(
            np.array(future_predictions_clean).reshape(-1, 1)
        ).flatten()
        predictions = np.clip(predictions, 0, None)
        
        # Generate dates
        last_ts = hourly.index[-1]
        future_dates = pd.date_range(
            start=last_ts + pd.Timedelta(hours=1),
            periods=future_hours, freq='h'
        )
        
        print(f"✅ Generated {len(predictions)} predictions")
        print(f"   From: {future_dates[0]}")
        print(f"   To: {future_dates[-1]}")
        print(f"   Mean: {predictions.mean():.1f} vehicles/hour")
        print(f"   Max: {predictions.max():.1f} vehicles/hour")
        print(f"   Min: {predictions.min():.1f} vehicles/hour")
        
        # Step 5: Plot results
        print(f"\n5️⃣  Creating visualization...")
        fig, ax = plt.subplots(figsize=(16, 7))
        
        # Historical data
        hist_hours = min(4*7*24, len(hourly))
        ax.plot(
            hourly.index[-hist_hours:],
            hourly['nombre_voitures'].tail(hist_hours),
            label='Historical Data',
            color='#2563EB', linewidth=1.5, alpha=0.8
        )
        
        # Predictions
        ax.plot(
            future_dates, predictions,
            label=f'LSTM Forecast ({future_hours}h)',
            color='#DC2626', linewidth=2, linestyle='--'
        )
        
        # Prediction zone
        ax.axvspan(future_dates[0], future_dates[-1], alpha=0.1, color='#DC2626')
        ax.axvline(x=last_ts, color='gray', linestyle=':', linewidth=2, alpha=0.7)
        
        ax.set_title(f'LSTM Traffic Forecast - {future_hours} hours', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Vehicles per hour', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save
        output_path = 'lstm_predictions_forecast.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved: {output_path}")
        plt.show()
        
        # Step 6: Return results
        print(f"\n✅ PREDICTION COMPLETE!")
        print("="*70 + "\n")
        
        return {
            'predictions': predictions,
            'dates': future_dates,
            'model': model,
            'scaler': scaler,
            'hourly_data': hourly,
            'lookback': lookback,
            'mean': predictions.mean(),
            'min': predictions.min(),
            'max': predictions.max()
        }
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run prediction with default parameters
    results = load_and_predict(
        model_path='lstm_keras_best.keras',
        lookback=23,
        future_hours=7*24,  # 7 days
        noise_std=0.05
    )
    
    if results:
        # Print prediction summary
        print("\n" + "="*70)
        print("PREDICTION SUMMARY")
        print("="*70)
        predictions_df = pd.DataFrame({
            'Date': results['dates'],
            'Predicted_Vehicles': results['predictions']
        })
        print(predictions_df.head(24))
        print(f"\n... ({len(predictions_df)-24} more rows)\n")




class PredictionRequest(BaseModel):
    """Request model for prediction parameters"""
    model_path: str = 'lstm_keras_best.keras'
    lookback: int = 23
    future_hours: int = 168  # 7 days
    noise_std: float = 0.05


@app.get("/", tags=["Info"])
async def root():
    """API information endpoint"""
    return {
        "name": "LSTM Traffic Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This info page",
            "POST /predict/json": "Get predictions as JSON",
            "POST /predict/plot": "Get plot as PNG image",
            "POST /predict/full": "Get predictions + plot as JSON"
        }
    }


@app.post("/predict/json", tags=["Predictions"])
async def predict_json(request: PredictionRequest):
    """
    Generate predictions and return as JSON
    
    Parameters:
    - model_path: Path to saved model (default: 'lstm_keras_best.keras')
    - lookback: Number of past hours to use (default: 23)
    - future_hours: Number of hours to predict (default: 168 = 7 days)
    - noise_std: Noise standard deviation (default: 0.05)
    """
    try:
        results = load_and_predict(
            model_path=request.model_path,
            lookback=request.lookback,
            future_hours=request.future_hours,
            noise_std=request.noise_std
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Format response
        predictions_list = [
            {
                "date": str(date),
                "predicted_vehicles": float(pred),
                "hour": i
            }
            for i, (date, pred) in enumerate(zip(results['dates'], results['predictions']))
        ]
        
        return {
            "status": "success",
            "metadata": {
                "mean_prediction": float(results['mean']),
                "min_prediction": float(results['min']),
                "max_prediction": float(results['max']),
                "total_predictions": len(predictions_list),
                "lookback_hours": results['lookback']
            },
            "predictions": predictions_list
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/plot", tags=["Predictions"])
async def predict_plot(request: PredictionRequest):
    """
    Generate predictions and return plot as PNG image
    """
    try:
        results = load_and_predict(
            model_path=request.model_path,
            lookback=request.lookback,
            future_hours=request.future_hours,
            noise_std=request.noise_std
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 7))
        
        # Historical data
        hist_hours = min(4*7*24, len(results['hourly_data']))
        ax.plot(
            results['hourly_data'].index[-hist_hours:],
            results['hourly_data']['nombre_voitures'].tail(hist_hours),
            label='Historical Data',
            color='#2563EB', linewidth=1.5, alpha=0.8
        )
        
        # Predictions
        ax.plot(
            results['dates'], results['predictions'],
            label=f'LSTM Forecast ({request.future_hours}h)',
            color='#DC2626', linewidth=2, linestyle='--'
        )
        
        # Formatting
        ax.axvspan(results['dates'][0], results['dates'][-1], alpha=0.1, color='#DC2626')
        ax.axvline(x=results['hourly_data'].index[-1], color='gray', linestyle=':', linewidth=2)
        ax.set_title(f'LSTM Traffic Forecast - {request.future_hours} hours', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Vehicles per hour', fontsize=12)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save to bytes
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plt.close()
        
        return FileResponse(
            buffer,
            media_type="image/png",
            filename="lstm_forecast.png"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/full", tags=["Predictions"])
async def predict_full(request: PredictionRequest):
    """
    Generate predictions and return both JSON + base64 encoded plot
    """
    try:
        results = load_and_predict(
            model_path=request.model_path,
            lookback=request.lookback,
            future_hours=request.future_hours,
            noise_std=request.noise_std
        )
        
        if not results:
            raise HTTPException(status_code=500, detail="Prediction failed")
        
        # Format predictions
        predictions_list = [
            {
                "date": str(date),
                "predicted_vehicles": float(pred),
                "hour": i
            }
            for i, (date, pred) in enumerate(zip(results['dates'], results['predictions']))
        ]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(16, 7))
        
        hist_hours = min(4*7*24, len(results['hourly_data']))
        ax.plot(
            results['hourly_data'].index[-hist_hours:],
            results['hourly_data']['nombre_voitures'].tail(hist_hours),
            label='Historical Data',
            color='#2563EB', linewidth=1.5, alpha=0.8
        )
        
        ax.plot(
            results['dates'], results['predictions'],
            label=f'LSTM Forecast ({request.future_hours}h)',
            color='#DC2626', linewidth=2, linestyle='--'
        )
        
        ax.axvspan(results['dates'][0], results['dates'][-1], alpha=0.1, color='#DC2626')
        ax.axvline(x=results['hourly_data'].index[-1], color='gray', linestyle=':', linewidth=2)
        ax.set_title(f'LSTM Traffic Forecast', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Vehicles per hour')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return {
            "status": "success",
            "metadata": {
                "mean_prediction": float(results['mean']),
                "min_prediction": float(results['min']),
                "max_prediction": float(results['max']),
                "total_predictions": len(predictions_list),
                "lookback_hours": results['lookback']
            },
            "predictions": predictions_list,
            "plot": f"data:image/png;base64,{plot_base64}"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
