# FastAPI LSTM Prediction - Postman Testing Guide

## 1. Start the API Server

```bash
cd C:\Users\user\Desktop\projet\test
python run_api.py
```

The API will start at: **http://localhost:8000**

## 2. API Endpoints

### 2.1 GET / - API Info
**URL:** `http://localhost:8000/`  
**Method:** `GET`  
**Description:** Get API information and available endpoints

**Response:**
```json
{
  "name": "LSTM Traffic Prediction API",
  "version": "1.0.0",
  "endpoints": {
    "GET /": "This info page",
    "POST /predict/json": "Get predictions as JSON",
    "POST /predict/plot": "Get plot as PNG image",
    "POST /predict/full": "Get predictions + plot as JSON"
  }
}
```

---

### 2.2 POST /predict/json - Predictions as JSON
**URL:** `http://localhost:8000/predict/json`  
**Method:** `POST`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "model_path": "lstm_keras_best.keras",
  "lookback": 23,
  "future_hours": 168,
  "noise_std": 0.05
}
```

**Response:**
```json
{
  "status": "success",
  "metadata": {
    "mean_prediction": 32.5,
    "min_prediction": 5.2,
    "max_prediction": 58.3,
    "total_predictions": 168,
    "lookback_hours": 23
  },
  "predictions": [
    {
      "date": "2024-04-15 08:00:00",
      "predicted_vehicles": 35.2,
      "hour": 0
    },
    {
      "date": "2024-04-15 09:00:00",
      "predicted_vehicles": 38.1,
      "hour": 1
    }
    // ... more predictions
  ]
}
```

---

### 2.3 POST /predict/plot - Plot as PNG Image
**URL:** `http://localhost:8000/predict/plot`  
**Method:** `POST`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "model_path": "lstm_keras_best.keras",
  "lookback": 23,
  "future_hours": 168,
  "noise_std": 0.05
}
```

**Response:** PNG image file as binary data  
**Response Headers:**
```
Content-Type: image/png
Content-Disposition: attachment; filename="lstm_forecast.png"
```

---

### 2.4 POST /predict/full - JSON + Base64 Plot
**URL:** `http://localhost:8000/predict/full`  
**Method:** `POST`  
**Content-Type:** `application/json`

**Request Body:**
```json
{
  "model_path": "lstm_keras_best.keras",
  "lookback": 23,
  "future_hours": 168,
  "noise_std": 0.05
}
```

**Response:**
```json
{
  "status": "success",
  "metadata": {
    "mean_prediction": 32.5,
    "min_prediction": 5.2,
    "max_prediction": 58.3,
    "total_predictions": 168,
    "lookback_hours": 23
  },
  "predictions": [
    {
      "date": "2024-04-15 08:00:00",
      "predicted_vehicles": 35.2,
      "hour": 0
    }
    // ... more predictions
  ],
  "plot": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAA..."
}
```

---

## 3. Postman Setup Instructions

### 3.1 Create a New Collection

1. Open Postman
2. Click **Collections** → **New Collection**
3. Name it: "LSTM Traffic Prediction API"

### 3.2 Add Requests

#### Request 1: Get API Info
- **Name:** Get API Info
- **Method:** GET
- **URL:** `{{base_url}}/`
- **Click Send**

#### Request 2: JSON Predictions
- **Name:** Get Predictions (JSON)
- **Method:** POST
- **URL:** `{{base_url}}/predict/json`
- **Headers:**
  - `Content-Type: application/json`
- **Body (raw):**
```json
{
  "model_path": "lstm_keras_best.keras",
  "lookback": 23,
  "future_hours": 168,
  "noise_std": 0.05
}
```

#### Request 3: Plot Image
- **Name:** Get Predictions (Plot)
- **Method:** POST
- **URL:** `{{base_url}}/predict/plot`
- **Headers:**
  - `Content-Type: application/json`
- **Body (raw):**
```json
{
  "model_path": "lstm_keras_best.keras",
  "lookback": 23,
  "future_hours": 168,
  "noise_std": 0.05
}
```

#### Request 4: Full Response (JSON + Plot)
- **Name:** Get Predictions (Full)
- **Method:** POST
- **URL:** `{{base_url}}/predict/full`
- **Headers:**
  - `Content-Type: application/json`
- **Body (raw):**
```json
{
  "model_path": "lstm_keras_best.keras",
  "lookback": 23,
  "future_hours": 336,
  "noise_std": 0.1
}
```

### 3.3 Setup Environment Variables

1. Click **Environments** → **Create new**
2. Name it: "Local"
3. Add variable:
   - **Variable:** `base_url`
   - **Value:** `http://localhost:8000`
4. Select this environment in the top-right dropdown

---

## 4. Test Different Scenarios

### Test 1: Short-term Forecast (24 hours)
```json
{
  "model_path": "lstm_keras_best.keras",
  "lookback": 23,
  "future_hours": 24,
  "noise_std": 0.05
}
```

### Test 2: Long-term Forecast (30 days)
```json
{
  "model_path": "lstm_keras_best.keras",
  "lookback": 23,
  "future_hours": 720,
  "noise_std": 0.03
}
```

### Test 3: High Variation
```json
{
  "model_path": "lstm_keras_best.keras",
  "lookback": 23,
  "future_hours": 168,
  "noise_std": 0.2
}
```

---

## 5. Interactive API Documentation

After starting the server, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

You can test all endpoints directly from the browser!

---

## 6. Curl Examples

### Get API Info
```bash
curl -X GET http://localhost:8000/
```

### Get JSON Predictions
```bash
curl -X POST http://localhost:8000/predict/json \
  -H "Content-Type: application/json" \
  -d '{"model_path":"lstm_keras_best.keras","lookback":23,"future_hours":168,"noise_std":0.05}'
```

### Save Plot Image
```bash
curl -X POST http://localhost:8000/predict/plot \
  -H "Content-Type: application/json" \
  -d '{"model_path":"lstm_keras_best.keras","lookback":23,"future_hours":168,"noise_std":0.05}' \
  --output forecast.png
```

---

## 7. Troubleshooting

### Port 8000 already in use
```bash
# Find process using port 8000
netstat -ano | findstr :8000

# Kill the process (replace PID)
taskkill /PID <PID> /F
```

### Database Connection Error
- Ensure PostgreSQL is running
- Check `.env` file has correct DB credentials

### Model Not Found
- Verify `lstm_keras_best.keras` exists in the test directory
- Check model path in request

---

## 8. Sample Full Response

```json
{
  "status": "success",
  "metadata": {
    "mean_prediction": 28.45,
    "min_prediction": 2.15,
    "max_prediction": 52.8,
    "total_predictions": 168,
    "lookback_hours": 23
  },
  "predictions": [
    {"date": "2024-04-15 08:00:00", "predicted_vehicles": 30.5, "hour": 0},
    {"date": "2024-04-15 09:00:00", "predicted_vehicles": 32.1, "hour": 1},
    {"date": "2024-04-15 10:00:00", "predicted_vehicles": 35.3, "hour": 2}
  ],
  "plot": "data:image/png;base64,iVBORw0KGgoAAAA..."
}
```
