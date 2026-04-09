import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from fastapi.responses import HTMLResponse
from app.database import engine

# Palette coherente (3 couleurs proches) pour les graphiques du dashboard
DASHBOARD_TONES = ['#1D4ED8', "#4C576E", "#F0F3F7"]


def _cycle_bar_colors(fig):
    """Apply a 3-tone cyclic palette to bar traces for visual consistency."""
    for trace in fig.data:
        if getattr(trace, 'type', None) == 'bar' and getattr(trace, 'x', None) is not None:
            n = len(trace.x)
            trace.marker.color = [DASHBOARD_TONES[i % len(DASHBOARD_TONES)] for i in range(n)]

def get_data():
    """
    Fetch fresh data from PostgreSQL database on every request.
    This ensures the dashboard always displays current data.
    """
    query = """
    SELECT 
        date_heure, gare, voie, classe_vehicule, 
        montant_paye, montant_net, type_paiement, 
        statut_abonnement, type_transaction
    FROM fact_transactions
    WHERE montant_net > 0
    """
    try:
        df = pd.read_sql(query, engine)
        if df.empty:
            raise ValueError("No data returned from database. Check if the table exists.")
        return df
    except Exception as e:
        print(f"Database error: {e}")
        raise

# ==================== PLOTS INTERACTIFS ====================

def plot_transactions_by_payment():
    df = get_data()
    payment_counts = df['type_paiement'].value_counts().reset_index()
    payment_counts.columns = ['Type de Paiement', 'Nombre de Transactions']
    
    fig = px.bar(
        payment_counts,
        x='Type de Paiement', 
        y='Nombre de Transactions',
        title="Nombre de Transactions par Type de Paiement",
        labels={'Type de Paiement': 'Type de Paiement', 'Nombre de Transactions': 'Nombre de Transactions'},
        color='Nombre de Transactions',
        text='Nombre de Transactions'
    )
    fig.update_layout(height=600, xaxis_tickangle=45)
    fig.update_traces(textposition='auto')
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
        color_discrete_sequence=DASHBOARD_TONES
    )
    _cycle_bar_colors(fig)
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
    try:
        df['date'] = pd.to_datetime(df['date_heure']).dt.date
    except Exception as e:
        print(f"Error parsing date_heure: {e}")
        df['date'] = df['date_heure'].astype(str).str[:10]
    
    daily = df.groupby('date')['montant_net'].sum().reset_index()
    daily = daily.sort_values('date')
    
    fig = px.line(
        daily, x='date', y='montant_net',
        title="Évolution Journalière du Chiffre d'Affaires",
        labels={'date': 'Date', 'montant_net': 'CA (FCFA)'},
        markers=True
    )
    fig.update_layout(height=600)
    return fig.to_html(full_html=False)


def plot_monthly_revenue_by_payment():
    df = get_data()
    df['date_heure'] = pd.to_datetime(df['date_heure'], errors='coerce')
    df = df.dropna(subset=['date_heure'])

    monthly = (
        df.assign(mois=df['date_heure'].dt.to_period('M').astype(str))
        .groupby(['mois', 'type_paiement'], as_index=False)['montant_net']
        .sum()
    )

    fig = px.bar(
        monthly,
        x='mois',
        y='montant_net',
        color='type_paiement',
        barmode='stack',
        title="CA Mensuel par Type de Paiement",
        labels={
            'mois': 'Mois',
            'montant_net': 'CA (FCFA)',
            'type_paiement': 'Type de Paiement'
        },
        color_discrete_sequence=DASHBOARD_TONES
    )
    fig.update_layout(height=600, xaxis_tickangle=-35)
    return fig.to_html(full_html=False)


def plot_transactions_by_hour():
    df = get_data()
    df['date_heure'] = pd.to_datetime(df['date_heure'], errors='coerce')
    df = df.dropna(subset=['date_heure'])
    df['heure'] = df['date_heure'].dt.hour

    hourly = (
        df.groupby('heure', as_index=False)
        .size()
        .rename(columns={'size': 'nombre_transactions'})
    )

    fig = px.bar(
        hourly,
        x='heure',
        y='nombre_transactions',
        title="Nombre de Transactions par Heure",
        labels={
            'heure': 'Heure de la Journée',
            'nombre_transactions': 'Nombre de Transactions'
        },
        color_discrete_sequence=DASHBOARD_TONES
    )
    _cycle_bar_colors(fig)
    fig.update_layout(height=600, xaxis=dict(dtick=1))
    return fig.to_html(full_html=False)


def plot_revenue_by_weekday():
    df = get_data()
    df['date_heure'] = pd.to_datetime(df['date_heure'], errors='coerce')
    df = df.dropna(subset=['date_heure'])

    day_map = {
        0: 'Lundi',
        1: 'Mardi',
        2: 'Mercredi',
        3: 'Jeudi',
        4: 'Vendredi',
        5: 'Samedi',
        6: 'Dimanche'
    }
    day_order = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

    df['jour_num'] = df['date_heure'].dt.dayofweek
    df['jour_semaine'] = df['jour_num'].map(day_map)

    weekly = (
        df.groupby(['jour_num', 'jour_semaine'], as_index=False)['montant_net']
        .sum()
        .sort_values('jour_num')
    )

    fig = px.bar(
        weekly,
        x='jour_semaine',
        y='montant_net',
        title="Chiffre d'Affaires par Jour de Semaine",
        labels={
            'jour_semaine': 'Jour de Semaine',
            'montant_net': 'CA (FCFA)'
        },
        color_discrete_sequence=DASHBOARD_TONES,
        category_orders={'jour_semaine': day_order}
    )
    _cycle_bar_colors(fig)
    fig.update_layout(height=600)
    return fig.to_html(full_html=False)