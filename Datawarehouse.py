import pandas as pd
from sqlalchemy import create_engine, text
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ====================== CONFIGURATION ======================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PostgreSQL Connection Settings (loaded from .env)
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "tollxpress_dw")
TABLE_NAME = os.getenv("TABLE_NAME", "fact_transactions")
CSV_FILE = os.getenv("CSV_FILE", "donnees_synthetiques_tollxpress_benin_2023-2024.csv")

# Create engine
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def extract(csv_file):
    logging.info(f"Extracting data from {csv_file}...")
    df = pd.read_csv(csv_file)
    logging.info(f"✅ Extracted {len(df):,} rows")
    return df

def transform(df):
    logging.info("Transforming data...")
    
    # Rename columns to snake_case
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    
    # Convert datetime
    df['date_heure'] = pd.to_datetime(df['date_heure'], errors='coerce')
    
    # Clean amount (remove negatives)
    df['montant_paye'] = pd.to_numeric(df['montant_paye'], errors='coerce')
    df['montant_net'] = df['montant_paye'].clip(lower=0)
    
    # Add load timestamp
    df['load_timestamp'] = datetime.now()
    
    logging.info("✅ Transformation completed")
    return df

def create_table(engine, table_name):
    """Create the main fact table"""
    create_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id_transaction      BIGINT PRIMARY KEY,
        date_heure          TIMESTAMP,
        gare                VARCHAR(100),
        voie                INTEGER,
        classe_vehicule     VARCHAR(50),
        montant_paye        NUMERIC(12,2),
        montant_net         NUMERIC(12,2),
        type_paiement       VARCHAR(50),
        statut_abonnement   VARCHAR(10),
        type_transaction    VARCHAR(50),
        load_timestamp      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    with engine.connect() as conn:
        conn.execute(text(create_sql))
    logging.info(f"✅ Table '{table_name}' created or already exists")

def load(df, engine, table_name):
    logging.info(f"Loading {len(df):,} rows into PostgreSQL table '{table_name}'...")
    
    # Load data (replace table each time - good for full refresh)
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists='replace',
        index=False,
        method='multi',
        chunksize=15000
    )
    logging.info("✅ Data loaded successfully")

def run_etl():
    start_time = datetime.now()
    logging.info("🚀 Starting ETL Pipeline - TollXpress Data Warehouse")

    # Run pipeline
    df = extract(CSV_FILE)
    df_clean = transform(df)
    create_table(engine, TABLE_NAME)
    load(df_clean, engine, TABLE_NAME)

    duration = datetime.now() - start_time
    logging.info(f"🎉 ETL Completed in {duration.total_seconds():.2f} seconds!")

if __name__ == "__main__":
    run_etl()