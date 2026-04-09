import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
from app.database import get_engine
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


def _load_classes_data() -> pd.DataFrame:
    """Charge les données par classe depuis la base de données"""
    query = """
    SELECT
        DATE_TRUNC('hour', date_heure) AS "date_heure",
        classe_vehicule AS "classe_vehicule",
        COUNT(id_transaction) AS "count"
    FROM fact_transactions
    GROUP BY DATE_TRUNC('hour', date_heure), classe_vehicule
    ORDER BY date_heure, classe_vehicule
    """
    df = pd.read_sql(query, get_engine())
    df["date_heure"] = pd.to_datetime(df["date_heure"])
    return df


def _calculate_hourly_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les patterns de répartition pour chaque heure de la journée (0-23).
    Retourne la proportion moyenne (%) par heure pour chaque classe.
    """
    df_hourly = df.copy()
    df_hourly["hour"] = df_hourly["date_heure"].dt.hour
    
    # Regrouper par heure et classe
    hourly_patterns = (
        df_hourly.groupby(["hour", "classe_vehicule"])["count"]
        .mean()
        .unstack(fill_value=0)
    )
    
    # Normaliser en proportions (%)
    hourly_proportions = hourly_patterns.div(hourly_patterns.sum(axis=1), axis=0) * 100
    
    return hourly_proportions


def _calculate_daily_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule les patterns de répartition pour chaque jour de la semaine.
    Retourne la proportion moyenne (%) par jour pour chaque classe.
    """
    df_daily = df.copy()
    df_daily["day_of_week"] = df_daily["date_heure"].dt.day_name()
    
    daily_patterns = (
        df_daily.groupby(["day_of_week", "classe_vehicule"])["count"]
        .mean()
        .unstack(fill_value=0)
    )
    
    daily_proportions = daily_patterns.div(daily_patterns.sum(axis=1), axis=0) * 100
    
    return daily_proportions

