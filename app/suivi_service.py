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
    
    # Calculate statistics for each vehicle class
    stats = df.groupby('classe_vehicule')['montant_net'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('std', 'std')
    ]).reset_index()
    
    # Create enhanced box plot
    fig = px.box(
        df, 
        x='classe_vehicule', 
        y='montant_net',
        points='outliers',  # Show outlier points
        title="Distribution des Montants par Classe de Véhicule",
        labels={
            'classe_vehicule': 'Classe Véhicule', 
            'montant_net': 'Montant Net (FCFA)'
        },
        color='classe_vehicule',
        color_discrete_map={
            'MOTO': '#EF4444',      # Red
            'VL': '#3B82F6',        # Blue
            'PL': '#10B981',        # Green
            'BUS': '#F59E0B'        # Amber
        }
    )
    
    # Add mean marker for each class
    for idx, row in stats.iterrows():
        fig.add_scatter(
            x=[row['classe_vehicule']],
            y=[row['mean']],
            mode='markers',
            marker=dict(size=12, symbol='diamond', color='yellow', 
                       line=dict(color='black', width=2)),
            name='Moyenne' if idx == 0 else '',
            hovertemplate=f"<b>{row['classe_vehicule']}</b><br>Moyenne: {row['mean']:.0f} FCFA<extra></extra>",
            showlegend=(idx == 0)
        )
    
    # Add annotations with statistics
    annotations = []
    vehicle_classes = df['classe_vehicule'].unique()
    
    for i, vehicle_class in enumerate(sorted(vehicle_classes)):
        class_stats = stats[stats['classe_vehicule'] == vehicle_class].iloc[0]
        
        annotation_text = (
            f"<b>{vehicle_class}</b><br>"
            f"Transactions: {int(class_stats['count'])}<br>"
            f"Moyenne: {class_stats['mean']:.0f} FCFA<br>"
            f"Médiane: {class_stats['median']:.0f} FCFA<br>"
            f"Min-Max: {class_stats['min']:.0f} - {class_stats['max']:.0f} FCFA"
        )
        
        annotations.append(dict(
            x=vehicle_class,
            y=class_stats['max'] * 1.15,  # Position above the box
            text=annotation_text,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#333",
            borderwidth=1,
            font=dict(size=9),
            align="center",
            xanchor="center"
        ))
    
    # Update layout with enhanced styling
    fig.update_layout(
        height=750,
        showlegend=False,
        template="plotly_white",
        hovermode="closest",
        xaxis=dict(
            title=dict(text="<b>Classe Véhicule</b>", font=dict(size=12, color="#333")),
            tickfont=dict(size=11),
            showgrid=True,
            gridwidth=1,
            gridcolor="#E5E7EB"
        ),
        yaxis=dict(
            title=dict(text="<b>Montant Net (FCFA)</b>", font=dict(size=12, color="#333")),
            tickfont=dict(size=10),
            showgrid=True,
            gridwidth=1,
            gridcolor="#E5E7EB",
            zeroline=True
        ),
        title=dict(
            text="<b>Distribution des Montants par Classe de Véhicule</b><br>" +
                 "<sub>Les points jaunes représentent la moyenne | Largeur de la boîte = écart interquartile</sub>",
            x=0.5,
            xanchor="center",
            font=dict(size=14)
        ),
        annotations=annotations,
        margin=dict(t=150, b=100, l=80, r=80)
    )
    
    # Update only the box traces for better visibility (don't update scatter traces)
    fig.update_traces(
        selector=dict(type="box"),
        boxmean=False,  # We're adding mean markers manually
        marker=dict(size=4, opacity=0.6),
        line=dict(width=2)
    )
    
    # Add a legend explanation at the bottom
    fig.add_annotation(
        text="📊 <b>Légende:</b> La boîte représente 50% des données | Ligne in the boîte = médiane | Points isolés = valeurs aberrantes",
        xref="paper", yref="paper",
        x=0.5, y=-0.15,
        showarrow=False,
        bgcolor="rgba(229, 231, 235, 0.5)",
        bordercolor="#999",
        borderwidth=1,
        font=dict(size=9),
        align="center",
        xanchor="center"
    )
    
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