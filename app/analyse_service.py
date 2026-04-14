import pandas as pd
import numpy as np
from prophet import Prophet
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app.database import get_engine
import warnings
import joblib as jb
warnings.filterwarnings("ignore")

def load() :
    
    query = """
    SELECT
        date_heure AS "Date_Heure",
        id_transaction AS "ID_transaction",
        gare AS "Gare",
        classe_vehicule AS "Classe_vehicule"
    FROM fact_transactions
    """
    df = pd.read_sql(query, get_engine())
    df["Date_Heure"] = pd.to_datetime(df["Date_Heure"])
    return df

def analyse_par_classe():
    """Get class analysis data as JSON"""
    df = load()
    df_classe = df.groupby([pd.Grouper(key="Date_Heure", freq="h"), "Classe_vehicule"]).size().reset_index(name="count")
    
    # Calculate statistics per class
    stats = df.groupby("Classe_vehicule").agg({
        "ID_transaction": "count",
        "Date_Heure": ["min", "max"]
    }).reset_index()
    stats.columns = ["classe", "total_transactions", "first_transaction", "last_transaction"]
    
    # Convert timestamps to ISO format strings
    stats["first_transaction"] = stats["first_transaction"].astype(str)
    stats["last_transaction"] = stats["last_transaction"].astype(str)
    
    # Get hourly breakdown
    hourly_data = {}
    for classe in df_classe["Classe_vehicule"].unique():
        classe_data = df_classe[df_classe["Classe_vehicule"] == classe]
        hourly_data[classe] = [
            {
                "datetime": str(row["Date_Heure"]),
                "count": int(row["count"])
            }
            for _, row in classe_data.iterrows()
        ]
    
    return {
        "summary": stats.to_dict("records"),
        "hourly": hourly_data
    }


def analyse_par_gare():
    df = load()
    df_gare = df.groupby([pd.Grouper(key="Date_Heure", freq="h"), "Gare"]).size().reset_index(name="count")
    
    # Calculate statistics per gare
    stats = df.groupby("Gare").agg({
        "ID_transaction": "count",
        "Date_Heure": ["min", "max"]
    }).reset_index()
    stats.columns = ["gare", "total_transactions", "first_transaction", "last_transaction"]
    
    # Convert timestamps to ISO format strings
    stats["first_transaction"] = stats["first_transaction"].astype(str)
    stats["last_transaction"] = stats["last_transaction"].astype(str)
    
    # Get hourly breakdown
    hourly_data = {}
    for gare in df_gare["Gare"].unique():
        gare_data = df_gare[df_gare["Gare"] == gare]
        hourly_data[gare] = [
            {
                "datetime": str(row["Date_Heure"]),
                "count": int(row["count"])
            }
            for _, row in gare_data.iterrows()
        ]
    
    return {
        "summary": stats.to_dict("records"),
        "hourly": hourly_data
    }


def analyse_global():
    df = load()
    df_global = df.groupby(pd.Grouper(key="Date_Heure", freq="h")).size().reset_index(name="count")
    
    # Calculate statistics
    total_transactions = len(df)
    date_range = {
        "start": str(df["Date_Heure"].min()),
        "end": str(df["Date_Heure"].max())
    }
    
    # Get hourly breakdown
    hourly_data = [
        {
            "datetime": str(row["Date_Heure"]),
            "count": int(row["count"])
        }
        for _, row in df_global.iterrows()
    ]
    
    return {
        "total_transactions": total_transactions,
        "date_range": date_range,
        "hourly": hourly_data
    }




