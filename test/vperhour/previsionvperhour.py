import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM, Bidirectional, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

print(f'TensorFlow : {tf.__version__}')
print(f'GPU disponible : {len(tf.config.list_physical_devices("GPU")) > 0}')

# ── 1. CHARGEMENT & PREPROCESSING ────────────────────────────────────────
df = pd.read_csv('donnees_synthetiques_tollxpress_benin_2023-2024.csv')
df = df[['Date_Heure']].copy()
df['Date_Heure'] = pd.to_datetime(df['Date_Heure'])

# Agrégation horaire
hourly = df.resample('h', on='Date_Heure').size().reset_index(name='nombre_voitures')
hourly.set_index('Date_Heure', inplace=True)

# Compléter les heures manquantes (evite les sauts temporels)
full_idx = pd.date_range(hourly.index.min(), hourly.index.max(), freq='h')
hourly   = hourly.reindex(full_idx, fill_value=0)

print(f'Données : {len(hourly)} heures')
print(f'Période : {hourly.index.min()} → {hourly.index.max()}')
print(f'Moyenne : {hourly["nombre_voitures"].mean():.1f} véhicules/heure')
print(f'Max     : {hourly["nombre_voitures"].max()} | Min : {hourly["nombre_voitures"].min()}')
hourly.head()

# ── 2. NORMALISATION & SEQUENCES ──────────────────────────────────────────
values = hourly['nombre_voitures'].values.reshape(-1, 1)

scaler      = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# Paramètres — identiques à ton notebook PyTorch
LOOKBACK     = 168     # 7 jours d'historique
TRAIN_RATIO  = 0.8
BATCH_SIZE   = 64
EPOCHS       = 10      # EarlyStopping arrêtera avant si nécessaire
NOISE_STD    = 0.8     # Standard deviation for Gaussian noise
FUTURE_HOURS = 168     # 7 jours de prévision

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i : i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled_data, LOOKBACK)

# Split temporel (jamais aléatoire sur une série temporelle)
train_size = int(len(X) * TRAIN_RATIO)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f'X_train : {X_train.shape} | y_train : {y_train.shape}')
print(f'X_test  : {X_test.shape}  | y_test  : {y_test.shape}')


model = Sequential([
    # Bloc 1 : BiLSTM — capture les patterns dans les 2 sens
    Bidirectional(
        LSTM(128, return_sequences=True,
             kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        input_shape=(LOOKBACK, 1)
    ),
    BatchNormalization(),
    Dropout(0.3),

    # Bloc 2 : LSTM compression
    LSTM(64, return_sequences=True),
    BatchNormalization(),
    Dropout(0.2),

    # Bloc 3 : LSTM résumé final
    LSTM(32, return_sequences=False),
    BatchNormalization(),
    Dropout(0.2),

    # Tête de régression
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(1)   # sortie scalaire
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae']
)

model.summary()

# ── 4. CALLBACKS ──────────────────────────────────────────────────────────
callbacks = [
    # Arrête si val_loss ne s'améliore plus après 10 époques
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    # Réduit le LR si plateau sur val_loss
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    # Sauvegarde le meilleur modèle
    ModelCheckpoint(
        'lstm_keras_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    )
]

# ── 5. ENTRAINEMENT ───────────────────────────────────────────────────────
print('Entraînement du modèle LSTM Keras...')

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),   # test set = validation
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=False,                       # ne pas mélanger une série temporelle
    verbose=1
)

# Récupérer train loss et test loss
train_losses = history.history['loss']
test_losses  = history.history['val_loss']
train_mae    = history.history['mae']
test_mae     = history.history['val_mae']
epochs_ran   = range(1, len(train_losses) + 1)

print(f'\nArrêt à l\'époque {len(train_losses)}')
print(f'Meilleur Train Loss : {min(train_losses):.6f}')
print(f'Meilleur Test Loss  : {min(test_losses):.6f}')

# ── 6. VISUALISATION TRAIN LOSS vs TEST LOSS ──────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

# (A) MSE Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(epochs_ran, train_losses, label='Train Loss', color='#2563EB', linewidth=2)
ax1.plot(epochs_ran, test_losses,  label='Test Loss',  color='#DC2626', linewidth=2, linestyle='--')
ax1.set_title('Train Loss vs Test Loss (MSE)', fontweight='bold')
ax1.set_xlabel('Époque')
ax1.set_ylabel('MSE')
ax1.legend()
ax1.grid(True, alpha=0.3)

# (B) MAE
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(epochs_ran, train_mae, label='Train MAE', color='#059669', linewidth=2)
ax2.plot(epochs_ran, test_mae,  label='Test MAE',  color='#D97706', linewidth=2, linestyle='--')
ax2.set_title('Train MAE vs Test MAE', fontweight='bold')
ax2.set_xlabel('Époque')
ax2.set_ylabel('MAE (normalisé)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ── 7. EVALUATION SUR LE TEST SET ─────────────────────────────────────────
y_pred_scaled = model.predict(X_test, verbose=0)
y_pred        = scaler.inverse_transform(y_pred_scaled).flatten()
y_true        = scaler.inverse_transform(y_test).flatten()

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

print(f'\n{"="*45}')
print(f'  MAE   : {mae:.2f}  véhicules/heure')
print(f'  RMSE  : {rmse:.2f}  véhicules/heure')
print(f'  MAPE  : {mape:.2f} %')
print(f'{"="*45}')

# Graphique prédictions vs réel
plt.figure(figsize=(16, 6))
n = min(500, len(y_true))
plt.plot(y_true[:n],  label='Réel',        color='#2563EB', linewidth=1.5, alpha=0.8)
plt.plot(y_pred[:n],  label='Prédit Keras', color='#DC2626', linewidth=1.5, linestyle='--')
plt.fill_between(range(n), y_true[:n], y_pred[:n], alpha=0.15, color='#7C3AED', label='Écart')
plt.title(f'Prédictions vs Réel — {n} premières heures du test  (MAE={mae:.1f} | RMSE={rmse:.1f} | MAPE={mape:.1f}%)',
          fontweight='bold')
plt.xlabel('Heure')
plt.ylabel('Nb véhicules')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lstm_keras_predictions.png', dpi=150, bbox_inches='tight')
plt.show()

# ── 9. VISUALISATION PRÉVISIONS FUTURES ───────────────────────────────────
HIST_HOURS = 4 * 7 * 24   # 4 semaines d'historique affiché

plt.figure(figsize=(18, 8))

# Historique réel
plt.plot(
    hourly.index[-HIST_HOURS:],
    hourly['nombre_voitures'].tail(HIST_HOURS),
    label='Historique réel',
    color='#2563EB', linewidth=1.5, alpha=0.75
)

# Prévisions avec bruit gaussien
plt.plot(
    future_dates, future_preds,
    label=f'Prévision LSTM Keras (bruit gaussien σ={NOISE_STD})',
    color='#DC2626', linewidth=2, linestyle='--'
)

# Zone prévision
plt.axvspan(future_dates[0], future_dates[-1],
            alpha=0.06, color='#DC2626', label='Zone prévue')

# Ligne de séparation
plt.axvline(x=last_ts, color='gray', linestyle=':', linewidth=1.5, label='Début prévision')

plt.title(f'Prévision LSTM Keras — {FUTURE_HOURS}h (7 jours) avec bruit gaussien σ={NOISE_STD}',
          fontweight='bold', fontsize=13)
plt.xlabel('Date')
plt.ylabel('Nombre de véhicules par heure')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lstm_keras_forecast.png', dpi=150, bbox_inches='tight')
plt.show()

