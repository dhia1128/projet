# TollXpress Bénin - Dashboard & Traffic Prediction

This project is an end-to-end data analytics, machine learning, and visualization platform for toll station traffic management in Benin.

## 🏗️ System Architecture

The system is composed of four main interconnected modules:

### 1. Data Warehouse & ETL (`Datawarehouse.py`)
- **Source:** CSV files containing synthetic toll transactions (2023-2024).
- **Processing:** Pandas is used to clean, format, and transform the data.
- **Storage:** Data is loaded into a **PostgreSQL** database (`tollxpress_dw`) via SQLAlchemy.

### 2. Backend API & Dashboard (`app/main.py`)
- **Framework:** Built with **FastAPI**.
- **Routing:** Serves HTML templates for the UI and exposes endpoints for data visualizations.
- **Database Connection:** Managed via SQLAlchemy (`app/database.py`).

### 3. Interactive Analytics (`app/plots_plotly.py`)
- **Engine:** Uses **Plotly Express** & **Graph Objects**.
- **Features:** Queries live data from PostgreSQL to generate real-time interactive charts, including:
  - Daily revenue evolution.
  - Top 10 toll stations by revenue.
  - Payment type distribution.
  - Revenue by vehicle class.

### 4. Machine Learning Forecasting (`main2.py`)
- **Engine:** Powered by **Scikit-Learn** (RandomForest Regressor).
- **Features:**
  - Advanced time-series feature engineering (cyclical time, lag features, peak hour flags).
  - Predicts global network traffic (vehicles/hour).
  - Predicts localized traffic per toll station.
  - Predicts vehicle class distribution.
  - Generates 7-day future forecasts (`previsions_J7_tollxpress.csv`).

## 🚀 Technologies Used

- **Python 3.x**
- **FastAPI** & Uvicorn (Web Server)
- **PostgreSQL** & SQLAlchemy (Database & ORM)
- **Pandas** & NumPy (Data Manipulation)
- **Plotly** (Data Visualization)
- **Scikit-Learn** (Machine Learning)

## ⚙️ Setup Instructions
1. Ensure PostgreSQL is running and update the `.env` variables.
2. Run the ETL pipeline: `python Datawarehouse.py`
3. Start the FastAPI server: `uvicorn app.main:app --reload`
4. Run the Machine Learning pipeline: `python main2.py`