# TollXpress - Système de Prévision de Trafic Routier

## 📋 Vue d'ensemble

**TollXpress** est une application intégrée de **prévision du trafic routier** et **analyse de données** pour la gestion des péages routiers au Bénin. Elle permet de visualiser les tendances du trafic, d'analyser les données de transaction et de générer des prévisions du nombre de véhicules.

---

## 🏗️ Architecture Globale

L'application suit une architecture **3-tiers** classique avec une séparation claire entre les couches:

```
┌─────────────────────────────────────────────────────────┐
│         Interface Web (FastAPI + Templates HTML)        │
│                   (Couche Présentation)                 │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│      Backend (Logique Métier + Services)                │
│  - Prévisions (Prophet, LSTM, Random Forest)           │
│  - Visualisations (Plotly)                              │
│  - Services de prédiction                               │
│         (Couche Application / Métier)                   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│   Données (PostgreSQL + ETL)                            │
│  - Base de Données (Data Warehouse)                     │
│  - Pipeline ETL (Extract, Transform, Load)              │
│         (Couche Données)                                │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Structure du Projet

```
projet/
├── app/                              # Cœur de l'application FastAPI
│   ├── __init__.py
│   ├── main.py                       # Point d'entrée FastAPI + endpoints
│   ├── database.py                   # Configuration PostgreSQL
│   ├── module_prevision.py           # Modèles de prévision (Prophet, RF)
│   ├── prediction_service.py         # Service de génération de prévisions
│   ├── plots_plotly.py               # Visualisations Plotly interactives
│   ├── pltos.py                      # Utilitaires graphiques
│   └── templates/                    # Pages HTML
│       ├── home.html                 # Page d'accueil
│       ├── dashboard.html            # Dashboard principal
│       └── predictions.html          # Page des prévisions
│
├── previsionvehicule/                # Modèles ML et notebooks
│   ├── lstm_heure.keras              # Modèle LSTM pré-entraîné
│   └── prevision.ipynb               # Notebook de prévision LSTM
│
├── src/                              # Données sources
│   └── donnees_synthetiques_tollxpress_benin_2023-2024.csv
│
├── Datawarehouse.py                  # Pipeline ETL + gestion du DW
├── EDA_ypnb.ipynb                    # Exploratory Data Analysis
├── requirements.txt                  # Dépendances Python
└── README.md                         # Ce fichier
```

---

## 🔧 Composants Techniques

### 1. **Backend API (FastAPI)**
- **Framework**: FastAPI (asynchrone, haute performance)
- **Serveur ASGI**: Uvicorn
- **Endpoints principaux**:
  - `/health` - Vérification de la connexion à la base de données
  - `/home` - Page d'accueil
  - `/plot/paiement` - Transactions par mode de paiement
  - `/plot/montant` - Distribution des montants
  - `/plot/ca_gare` - CA par gare
  - Des endpoints supplémentaires pour les prévisions

### 2. **Couche Données**
- **SGBD**: PostgreSQL
- **ORM**: SQLAlchemy
- **Driver**: psycopg2
- **Table principale**: `fact_transactions` (transactions de péage)
- **Colonnes clés**: Gare, Date_Heure, Montant_Paye, Classe_Vehicule

### 3. **Pipeline ETL (Datawarehouse.py)**
Processus d'importation et de transformation des données:

**Extract** → Lecture du fichier CSV  
**Transform** → Nettoyage, conversion de types, extraction d'features  
**Load** → Insertion dans PostgreSQL  

Transformations appliquées:
- Normalisation des colonnes (snake_case)
- Conversion des datetime
- Nettoyage des montants (suppression des négatifs)
- Ajout d'timestamps de chargement

### 4. **Module de Prévision (module_prevision.py)**
Utilise plusieurs algorithmes:
- **Prophet**: Prévision de séries temporelles avec gestion des tendances et saisonnalité
- **Random Forest Regressor**: Prévision du nombre de véhicules par classe
- **LSTM (Keras)**: Modèle de deep learning pour prévisions à court terme (dans `previsionvehicule/lstm_heure.keras`)

### 5. **Visualisations (plots_plotly.py)**
Graphiques interactifs Plotly:
- Distribution des paiements
- Distribution des montants
- Classement des gares par CA
- Tendances quotidiennes du trafic
- Prévisions visuelles

### 6. **Analyse Exploratoire (EDA_ypnb.ipynb)**
Notebook Jupyter pour:
- Analyse statistique des données
- Distributions et corrélations
- Identification des patterns et anomalies
- Sélection des features

---

## 🔄 Flux de Données

```
CSV (Source)
    │
    ▼
Datawarehouse.py (ETL)
    │
    ├─ Extract ─► Lecture CSV
    ├─ Transform ─► Nettoyage
    └─ Load ─► PostgreSQL
    │
    ▼
Base de Données (PostgreSQL)
    │
    ├─► module_prevision.py (Modèles ML)
    ├─► plots_plotly.py (Graphiques)
    └─► prediction_service.py (Service)
    │
    ▼
FastAPI (main.py)
    │
    ▼
Interface Web (HTML + Plotly)
```

---

## 📊 Modèles de Prévision

| Modèle | Utilisation | Avantages | Limitations |
|--------|-------------|-----------|------------|
| **Prophet** | Prévisions à moyen terme | Gère tendances & saisonnalité | Configuration manuelle |
| **Random Forest** | Classification par classe véhicule | Non-linéaire, robuste | Consommation mémoire |
| **LSTM** | Prévisions à court terme (horaires) | Apprend les dépendances tempor. | Nécessite plus de données |

---

## 🛠️ Stack Technologique

| Couche | Technologie | Version |
|--------|-------------|---------|
| **Backend** | FastAPI | 0.104.1 |
| **Serveur** | Uvicorn | 0.24.0 |
| **BD** | PostgreSQL | - |
| **ORM** | SQLAlchemy | 2.0.23 |
| **Données** | Pandas, NumPy | 2.0.3, 2.0.3 |
| **ML** | scikit-learn, Prophet | 1.3.0, 1.1.5 |
| **DL** | Keras/TensorFlow | (dans .keras) |
| **Visualisation** | Plotly | 5.17.0 |
| **Tests** | pytest | 7.3.1 |
| **Migration BD** | Alembic | 1.7.7 |

---

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.8+
- PostgreSQL installé et configuré
- Variables d'environnement dans `.env`:
  ```
  DB_USER=postgres
  DB_PASSWORD=votre_password
  DB_HOST=127.0.0.1
  DB_PORT=5432
  DB_NAME=tollxpress_dw
  CSV_FILE=src/donnees_synthetiques_tollxpress_benin_2023-2024.csv
  ```

### Installation
```bash
# Installation des dépendances
pip install -r requirements.txt

# Pipeline ETL (charger les données)
python Datawarehouse.py

# Démarrer l'application
uvicorn app.main:app --reload
```

**Accès**: http://localhost:8000/home

---

## 📈 Cas d'Usage

1. **Analyse du Trafic**: Visualiser les tendances historiques du trafic par gare et par heure
2. **Prévisions de Revenus**: Anticiper les revenus par classe de véhicule
3. **Optimisation des Ressources**: Planifier le déploiement du personnel selon les pics de trafic
4. **Détection d'Anomalies**: Identifier les variations anormales du trafic
5. **Reporting**: Générer des rapports visuels interactifs

---

## 📝 Notes d'Architecture

- **Scalabilité**: Architecture modulaire permettant l'ajout de nouveaux modèles de prévision
- **Réutilisabilité**: Services métier séparés pour faciliter l'intégration
- **Maintenance**: Séparation claire entre ETL, logique métier et présentation
- **Extensibilité**: Support des notebooks Jupyter pour la recherche et le prototypage

---

## 📚 Documentation Supplémentaire

- Modèles ML: voir `previsionvehicule/prevision.ipynb`
- Analyse des données: voir `EDA_ypnb.ipynb`
- Configuration BD: voir `app/database.py`

---

**Version**: 1.0  
**Auteur**: TollXpress Team  
**Dernière mise à jour**: 2024
