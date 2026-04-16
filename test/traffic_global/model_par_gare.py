import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')
import joblib as jb
import os

print(f'TensorFlow : {tf.__version__}')
print(f'GPU disponible : {len(tf.config.list_physical_devices("GPU")) > 0}')

# ── 1. CHARGEMENT & PREPROCESSING ────────────────────────────────────────
df_raw = pd.read_csv('../../src/donnees_synthetiques_tollxpress_benin_2023-2024.csv')
df_raw["Date_Heure"] = pd.to_datetime(df_raw["Date_Heure"])

# Obtenir les stations uniques
stations = df_raw["Gare"].unique()
print(f"\n📍 Stations trouvées: {sorted(stations)}\n")

# ── 2. PREPROCESSING PAR GARE ────────────────────────────────────────
def prepare_data_for_station(df, station, lookback=168):
    """Prépare les données pour une station spécifique"""
    
    # Filtrer par station
    df_station = df[df["Gare"] == station].copy()
    
    # Agréger par heure
    df_station["date_h"] = df_station["Date_Heure"].dt.floor("h")
    agg = df_station.groupby("date_h").size().rename("y")
    agg.index.name = "ds"
    hourly = agg.to_frame()
    
    # Compléter les heures manquantes
    full_idx = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full_idx, fill_value=0)
    hourly.index.name = "ds"
    
    # Normalisation
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(hourly[["y"]])
    
    # Créer séquences
    X, y = [], []
    for i in range(len(data_scaled) - lookback):
        X.append(data_scaled[i:i+lookback])
        y.append(data_scaled[i+lookback])
    
    X, y = np.array(X), np.array(y)
    
    # Split train/val/test: 70-15-15
    train_size = int(len(X) * 0.70)
    val_size = int(len(X) * 0.15)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
    
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'scaler': scaler,
        'hourly': hourly,
        'lookback': lookback,
        'n_samples': len(X),
    }

# ── 3. CRÉER & ENTRAÎNER MODÈLE PAR GARE ────────────────────────────────────────
def build_lstm_model(lookback=168):
    """Construit un modèle LSTM bidirectionnel"""
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, activation='relu'), 
                     input_shape=(lookback, 1)),
        Dropout(0.2),
        BatchNormalization(),
        
        Bidirectional(LSTM(32, return_sequences=False, activation='relu')),
        Dropout(0.2),
        BatchNormalization(),
        
        Dense(16, activation='relu'),
        Dropout(0.1),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# Créer dossier pour les modèles
os.makedirs('models_par_gare', exist_ok=True)

# Entraîner un modèle par gare
results_per_station = {}

for station in sorted(stations):
    print(f"\n{'='*60}")
    print(f"🚗 Traitement de la gare: {station}")
    print(f"{'='*60}")
    
    try:
        # Préparation des données
        data = prepare_data_for_station(df_raw, station, lookback=168)
        
        print(f"  Samples: {data['n_samples']}")
        print(f"  Train: {len(data['X_train'])}, Val: {len(data['X_val'])}, Test: {len(data['X_test'])}")
        
        # Construire le modèle
        model = build_lstm_model(lookback=168)
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001),
            ModelCheckpoint(f'models_par_gare/{station}_best.keras', 
                          monitor='val_loss', save_best_only=True),
        ]
        
        # Entraînement
        model.summary()
        history = model.fit(
            data['X_train'], data['y_train'],
            validation_data=(data['X_val'], data['y_val']),
            epochs=10,
            batch_size=32,
            verbose=0,
            callbacks=callbacks
        )
        
        # Évaluation sur test
        y_pred_test = model.predict(data['X_test'], verbose=0)
        
        mae = mean_absolute_error(data['y_test'], y_pred_test)
        rmse = np.sqrt(mean_squared_error(data['y_test'], y_pred_test))
        mape = mean_absolute_percentage_error(data['y_test'], y_pred_test)
        
        # Inverse transform pour valores reais
        y_test_actual = data['scaler'].inverse_transform(data['y_test'])
        y_pred_actual = data['scaler'].inverse_transform(y_pred_test)
        
        mae_actual = mean_absolute_error(y_test_actual, y_pred_actual)
        rmse_actual = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        
        print(f"  ✓ Modèle entraîné avec succès")
        print(f"  📊 MAE (normalisé): {mae:.6f}")
        print(f"  📊 RMSE (normalisé): {rmse:.6f}")
        print(f"  📊 MAPE: {mape:.2%}")
        print(f"  📊 MAE (véhicules/h): {mae_actual:.2f}")
        print(f"  📊 RMSE (véhicules/h): {rmse_actual:.2f}")
        
        # Sauvegarder le modèle et le scaler
        model.save(f'models_par_gare/{station}_model.keras')
        jb.dump(data['scaler'], f'models_par_gare/{station}_scaler.pkl')
        
        results_per_station[station] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mae_actual': mae_actual,
            'rmse_actual': rmse_actual,
            'samples': data['n_samples'],
        }
        
    except Exception as e:
        print(f"  ❌ Erreur: {e}")
        results_per_station[station] = {'error': str(e)}

# ── 4. RÉSUMÉ DES RÉSULTATS ────────────────────────────────────────
print(f"\n\n{'='*60}")
print(f"📈 RÉSUMÉ DES RÉSULTATS")
print(f"{'='*60}")

results_df = pd.DataFrame(results_per_station).T
print(results_df.to_string())

# Sauvegarder résumé
results_df.to_csv('results_par_gare.csv')
print(f"\n✓ Résultats sauvegardés dans 'results_par_gare.csv'")
print(f"✓ Modèles sauvegardés dans le dossier 'models_par_gare/'")
