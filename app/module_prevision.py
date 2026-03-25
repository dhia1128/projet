import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
import joblib
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from sqlalchemy import create_engine

# Load environment variables from .env file
load_dotenv()

# Build connection URL from environment variables
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "tollxpress_dw")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL, echo=False)
# Remplacez par vos informations de connexion

query="""SELECT 
    Gare,
    Heure,
    COUNT(*) AS "Nombre de véhicules observés",
    ROUND(AVG(COUNT(*)) OVER (PARTITION BY Gare, Heure), 2) AS "Moyenne par heure et gare"
FROM (
    SELECT 
        Gare,
        EXTRACT(HOUR FROM Date_Heure)::INTEGER AS Heure
    FROM fact_transactions
) 
GROUP BY Gare, Heure
ORDER BY Gare, Heure;
"""

df = pd.read_sql(query, engine)
print(df)

query2 = """SELECT count(id_transaction), classe_vehicule FROM public.fact_transactions
group by classe_vehicule ;"""

df2 = pd.read_sql(query2, engine)
print(df2)


