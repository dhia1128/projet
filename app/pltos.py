import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fastapi.responses import HTMLResponse

# Cache simple pour éviter de recharger les données à chaque fois
df_cache = None

def get_data():
    global df_cache
    if df_cache is None:
        query = """
        SELECT 
            date_heure, gare, voie, classe_vehicule, 
            montant_paye, montant_net, type_paiement, 
            statut_abonnement, type_transaction
        FROM fact_transactions
        WHERE montant_net > 0
        LIMIT 150000
        """
        df_cache = pd.read_sql(query, engine)
    return df_cache.copy()

# ==================== PLOTS INTERACTIFS ====================

def plot_transactions_by_payment():
    df = get_data()
    fig = px.bar(
        df['type_paiement'].value_counts().reset_index(),
        x='index', 
        y='type_paiement',
        title="Nombre de Transactions par Type de Paiement",
        labels={'index': 'Type de Paiement', 'type_paiement': 'Nombre de Transactions'},
        color='type_paiement',
        text='type_paiement'
    )
    fig.update_layout(height=600, xaxis_tickangle=45)
    return fig.to_html(full_html=False)

def plot_montant_distribution():
    df = get_data()
    fig = px.histogram(
        df, x='montant_net', nbins=80, 
        title="Distribution des Montants Payés",
        labels={'montant_net': 'Montant Net (FCFA)'},
        color_discrete_sequence=['#636EFA']
    )
    fig.update_layout(height=600, bargap=0.1)
    return fig.to_html(full_html=False)

def plot_top_gares_ca():
    df = get_data()
    top_gares = df.groupby('gare')['montant_net'].sum().nlargest(10).reset_index()
    
    fig = px.bar(
        top_gares, 
        x='montant_net', 
        y='gare',
        orientation='h',
        title="Top 10 des Gares par Chiffre d'Affaires",
        labels={'montant_net': 'Chiffre d\'Affaires (FCFA)', 'gare': 'Gare'},
        color='montant_net',
        color_continuous_scale='Viridis'
    )
    fig.update_layout(height=650)
    return fig.to_html(full_html=False)

def plot_montant_by_vehicle():
    df = get_data()
    fig = px.box(
        df, 
        x='classe_vehicule', 
        y='montant_net',
        title="Distribution des Montants par Classe de Véhicule",
        labels={'classe_vehicule': 'Classe Véhicule', 'montant_net': 'Montant Net (FCFA)'}
    )
    fig.update_layout(height=600)
    return fig.to_html(full_html=False)

def plot_daily_trend():
    df = get_data()
    df['date'] = pd.to_datetime(df['date_heure']).dt.date
    daily = df.groupby('date')['montant_net'].sum().reset_index()
    
    fig = px.line(
        daily, x='date', y='montant_net',
        title="Évolution Journalière du Chiffre d'Affaires",
        labels={'date': 'Date', 'montant_net': 'CA (FCFA)'},
        markers=True
    )
    fig.update_layout(height=600)
    return fig.to_html(full_html=False)