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
    df = load()
    df_classe = df.groupby([pd.Grouper(key="Date_Heure", freq="h"), "Classe_vehicule"]).size().reset_index(name="y")
    fig = go.Figure()
    for classe in df_classe["Classe_vehicule"].unique():
        classe_data = df_classe[df_classe["Classe_vehicule"] == classe]
        fig.add_trace(go.Scatter(x=classe_data["Date_Heure"], y=classe_data["y"], mode="lines", name=f"Classe {classe}"))
    fig.update_layout(title="Nombre de transactions par classe de véhicule", xaxis_title="Date et Heure", yaxis_title="Nombre de transactions")
    return fig.to_html(full_html=False)

def analyse_par_gare():
    df = load()
    df_gare = df.groupby([pd.Grouper(key="Date_Heure", freq="h"), "Gare"]).size().reset_index(name="y")
    fig = go.Figure()
    for gare in df_gare["Gare"].unique():
        gare_data = df_gare[df_gare["Gare"] == gare]
        fig.add_trace(go.Scatter(x=gare_data["Date_Heure"], y=gare_data["y"], mode="lines", name=f"Gare {gare}"))
    fig.update_layout(title="Nombre de transactions par gare", xaxis_title="Date et Heure", yaxis_title="Nombre de transactions")
    return fig.to_html(full_html=False)

def analyse_global():
    df = load()
    df_global = df.groupby(pd.Grouper(key="Date_Heure", freq="h")).size().reset_index(name="y")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_global["Date_Heure"], y=df_global["y"], mode="lines", name="Global"))
    fig.update_layout(title="Nombre total de transactions", xaxis_title="Date et Heure",
                        yaxis_title="Nombre de transactions")
    return fig.to_html(full_html=False)




