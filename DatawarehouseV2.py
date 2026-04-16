import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ====================== CONFIGURATION ======================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PostgreSQL Connection
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "tollxpress_dw")
CSV_FILE = os.getenv("CSV_FILE", "src/donnees_synthetiques_tollxpress_benin_2023-2024.csv")

engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# ====================== DIMENSION TABLES ======================

def create_dim_date_table(engine):
    """Create Date Dimension Table"""
    sql = """
    CREATE TABLE IF NOT EXISTS dim_date (
        date_id INTEGER PRIMARY KEY,
        date DATE NOT NULL,
        year INTEGER,
        quarter INTEGER,
        month INTEGER,
        month_name VARCHAR(20),
        day INTEGER,
        day_of_week INTEGER,
        day_name VARCHAR(20),
        week_of_year INTEGER,
        is_weekend BOOLEAN,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_dim_date_date ON dim_date(date);
    """
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info("✓ DIM_DATE table created")

def create_dim_station_table(engine):
    """Create Station Dimension Table"""
    sql = """
    CREATE TABLE IF NOT EXISTS dim_station (
        station_id SERIAL PRIMARY KEY,
        station_code VARCHAR(50) NOT NULL UNIQUE,
        station_name VARCHAR(100),
        region VARCHAR(100),
        country VARCHAR(50),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_dim_station_code ON dim_station(station_code);
    """
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info("✓ DIM_STATION table created")

def create_dim_vehicle_class_table(engine):
    """Create Vehicle Class Dimension Table"""
    sql = """
    CREATE TABLE IF NOT EXISTS dim_vehicle_class (
        vehicle_class_id SERIAL PRIMARY KEY,
        vehicle_class_code VARCHAR(50) NOT NULL UNIQUE,
        vehicle_class_name VARCHAR(100),
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_dim_vehicle_class ON dim_vehicle_class(vehicle_class_code);
    """
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info("✓ DIM_VEHICLE_CLASS table created")

def create_dim_payment_type_table(engine):
    """Create Payment Type Dimension Table"""
    sql = """
    CREATE TABLE IF NOT EXISTS dim_payment_type (
        payment_type_id SERIAL PRIMARY KEY,
        payment_type_code VARCHAR(50) NOT NULL UNIQUE,
        payment_type_name VARCHAR(100),
        description TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    
    CREATE INDEX IF NOT EXISTS idx_dim_payment_type ON dim_payment_type(payment_type_code);
    """
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info("✓ DIM_PAYMENT_TYPE table created")

# ====================== FACT TABLE ======================

def create_fact_transactions_table(engine):
    """Create Main Fact Table with Foreign Keys"""
    sql = """
    CREATE TABLE IF NOT EXISTS fact_transactions (
        transaction_id BIGINT PRIMARY KEY,
        date_id INTEGER NOT NULL,
        station_id INTEGER NOT NULL,
        vehicle_class_id INTEGER NOT NULL,
        payment_type_id INTEGER NOT NULL,
        lane_number INTEGER,
        amount_paid NUMERIC(12,2),
        amount_net NUMERIC(12,2),
        subscription_status VARCHAR(50),
        transaction_type VARCHAR(50),
        load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        -- Foreign Key Constraints
        FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
        FOREIGN KEY (station_id) REFERENCES dim_station(station_id),
        FOREIGN KEY (vehicle_class_id) REFERENCES dim_vehicle_class(vehicle_class_id),
        FOREIGN KEY (payment_type_id) REFERENCES dim_payment_type(payment_type_id)
    );
    
    -- Performance Indexes
    CREATE INDEX IF NOT EXISTS idx_fact_date_id ON fact_transactions(date_id);
    CREATE INDEX IF NOT EXISTS idx_fact_station_id ON fact_transactions(station_id);
    CREATE INDEX IF NOT EXISTS idx_fact_vehicle_id ON fact_transactions(vehicle_class_id);
    CREATE INDEX IF NOT EXISTS idx_fact_payment_id ON fact_transactions(payment_type_id);
    CREATE INDEX IF NOT EXISTS idx_fact_date_station ON fact_transactions(date_id, station_id);
    """
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info("✓ FACT_TRANSACTIONS table created")

# ====================== AGGREGATE TABLES ======================

def create_agg_daily_by_station_table(engine):
    """Create Daily Aggregate Table by Station"""
    sql = """
    CREATE TABLE IF NOT EXISTS agg_daily_by_station (
        agg_id SERIAL PRIMARY KEY,
        date_id INTEGER NOT NULL,
        station_id INTEGER NOT NULL,
        total_transactions BIGINT,
        total_amount NUMERIC(15,2),
        avg_amount NUMERIC(12,2),
        min_amount NUMERIC(12,2),
        max_amount NUMERIC(12,2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
        FOREIGN KEY (station_id) REFERENCES dim_station(station_id),
        UNIQUE(date_id, station_id)
    );
    
    CREATE INDEX IF NOT EXISTS idx_agg_daily_date ON agg_daily_by_station(date_id);
    CREATE INDEX IF NOT EXISTS idx_agg_daily_station ON agg_daily_by_station(station_id);
    """
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info("✓ AGG_DAILY_BY_STATION table created")

def create_agg_hourly_traffic_table(engine):
    """Create Hourly Aggregate Table for Traffic Analysis"""
    sql = """
    CREATE TABLE IF NOT EXISTS agg_hourly_traffic (
        hour_id SERIAL PRIMARY KEY,
        date_id INTEGER NOT NULL,
        hour INTEGER,
        total_vehicles BIGINT,
        total_revenue NUMERIC(15,2),
        avg_revenue NUMERIC(12,2),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        FOREIGN KEY (date_id) REFERENCES dim_date(date_id),
        UNIQUE(date_id, hour)
    );
    
    CREATE INDEX IF NOT EXISTS idx_agg_hourly_date ON agg_hourly_traffic(date_id);
    CREATE INDEX IF NOT EXISTS idx_agg_hourly_hour ON agg_hourly_traffic(hour);
    """
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info("✓ AGG_HOURLY_TRAFFIC table created")

# ====================== ETL FUNCTIONS ======================

def extract(csv_file):
    """Extract data from CSV"""
    logger.info(f"Extracting data from {csv_file}...")
    df = pd.read_csv(csv_file)
    logger.info(f"Extracted {len(df):,} rows")
    return df

def transform(df):
    """Transform and clean data"""
    logger.info("Transforming data...")
    
    # Standardize column names
    df.columns = [col.strip().lower().replace(' ', '_').replace('-', '_') for col in df.columns]
    
    # Convert datetime
    df['date_heure'] = pd.to_datetime(df['date_heure'], errors='coerce')
    
    # Clean numeric fields
    df['montant_paye'] = pd.to_numeric(df['montant_paye'], errors='coerce')
    df['montant_net'] = df['montant_paye'].clip(lower=0)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['id_transaction'])
    
    logger.info(f"Transformation completed - {len(df):,} rows")
    return df

def load_dimensions(df, engine):
    """Load Dimension Tables"""
    logger.info("Loading dimension tables...")
    
    # DIM_STATION
    stations = df[['gare']].drop_duplicates().rename(columns={'gare': 'station_code'})
    stations['station_name'] = stations['station_code']
    stations['region'] = 'Benin'
    stations['country'] = 'Benin'
    
    sql_delete_station = "DELETE FROM dim_station"
    with engine.connect() as conn:
        conn.execute(text(sql_delete_station))
        conn.commit()
    
    stations.to_sql('dim_station', engine, if_exists='append', index=False)
    logger.info(f"  ✓ Loaded {len(stations)} stations")
    
    # DIM_VEHICLE_CLASS
    vehicle_classes = df[['classe_vehicule']].drop_duplicates().rename(
        columns={'classe_vehicule': 'vehicle_class_code'}
    )
    vehicle_classes['vehicle_class_name'] = vehicle_classes['vehicle_class_code']
    
    sql_delete_vc = "DELETE FROM dim_vehicle_class"
    with engine.connect() as conn:
        conn.execute(text(sql_delete_vc))
        conn.commit()
    
    vehicle_classes.to_sql('dim_vehicle_class', engine, if_exists='append', index=False)
    logger.info(f"  ✓ Loaded {len(vehicle_classes)} vehicle classes")
    
    # DIM_PAYMENT_TYPE
    payment_types = df[['type_paiement']].drop_duplicates().rename(
        columns={'type_paiement': 'payment_type_code'}
    )
    payment_types['payment_type_name'] = payment_types['payment_type_code']
    
    sql_delete_pt = "DELETE FROM dim_payment_type"
    with engine.connect() as conn:
        conn.execute(text(sql_delete_pt))
        conn.commit()
    
    payment_types.to_sql('dim_payment_type', engine, if_exists='append', index=False)
    logger.info(f"  ✓ Loaded {len(payment_types)} payment types")

def load_date_dimension(engine):
    """Populate Date Dimension"""
    logger.info("Populating DIM_DATE...")
    
    sql = """
    WITH date_range AS (
        SELECT date_trunc('day', d)::date as date
        FROM generate_series('2023-01-01'::date, '2024-12-31'::date, '1 day'::interval) d
    )
    INSERT INTO dim_date (date_id, date, year, quarter, month, month_name, day, day_of_week, day_name, week_of_year, is_weekend)
    SELECT 
        ROW_NUMBER() OVER (ORDER BY date) as date_id,
        date,
        EXTRACT(YEAR FROM date)::INTEGER as year,
        EXTRACT(QUARTER FROM date)::INTEGER as quarter,
        EXTRACT(MONTH FROM date)::INTEGER as month,
        TO_CHAR(date, 'Month') as month_name,
        EXTRACT(DAY FROM date)::INTEGER as day,
        EXTRACT(DOW FROM date)::INTEGER as day_of_week,
        TO_CHAR(date, 'Day') as day_name,
        EXTRACT(WEEK FROM date)::INTEGER as week_of_year,
        EXTRACT(DOW FROM date) IN (0, 6) as is_weekend
    FROM date_range
    ON CONFLICT (date_id) DO NOTHING;
    """
    
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    logger.info("  ✓ DIM_DATE populated")

def load_fact_table(df, engine):
    """Load Fact Table"""
    logger.info("Loading FACT_TRANSACTIONS...")
    
    # Join with dimension IDs
    query = """
    SELECT 
        ft.id_transaction,
        dd.date_id,
        ds.station_id,
        dvc.vehicle_class_id,
        dpt.payment_type_id,
        ft.voie,
        ft.montant_paye,
        ft.montant_net,
        ft.statut_abonnement,
        ft.type_transaction,
        CURRENT_TIMESTAMP as load_timestamp
    FROM (
        SELECT * FROM fact_transactions_temp
    ) ft
    LEFT JOIN dim_date dd ON dd.date = ft.date_heure::date
    LEFT JOIN dim_station ds ON ds.station_code = ft.gare
    LEFT JOIN dim_vehicle_class dvc ON dvc.vehicle_class_code = ft.classe_vehicule
    LEFT JOIN dim_payment_type dpt ON dpt.payment_type_code = ft.type_paiement
    """
    
    # First, load to temp table
    df_temp = df[['id_transaction', 'date_heure', 'gare', 'classe_vehicule', 'type_paiement', 
                   'voie', 'montant_paye', 'montant_net', 'statut_abonnement', 'type_transaction']].copy()
    df_temp.columns = ['id_transaction', 'date_heure', 'gare', 'classe_vehicule', 'type_paiement',
                       'voie', 'montant_paye', 'montant_net', 'statut_abonnement', 'type_transaction']
    
    sql_delete_fact = "DELETE FROM fact_transactions"
    with engine.connect() as conn:
        conn.execute(text(sql_delete_fact))
        conn.commit()
    
    df_temp.to_sql('fact_transactions_temp', engine, if_exists='replace', index=False)
    
    # Insert from temp with proper joins
    insert_sql = """
    INSERT INTO fact_transactions (transaction_id, date_id, station_id, vehicle_class_id, payment_type_id, 
                                   lane_number, amount_paid, amount_net, subscription_status, transaction_type)
    SELECT 
        ft.id_transaction,
        dd.date_id,
        ds.station_id,
        dvc.vehicle_class_id,
        dpt.payment_type_id,
        ft.voie,
        ft.montant_paye,
        ft.montant_net,
        ft.statut_abonnement,
        ft.type_transaction
    FROM fact_transactions_temp ft
    LEFT JOIN dim_date dd ON dd.date = ft.date_heure::date
    LEFT JOIN dim_station ds ON ds.station_code = ft.gare
    LEFT JOIN dim_vehicle_class dvc ON dvc.vehicle_class_code = ft.classe_vehicule
    LEFT JOIN dim_payment_type dpt ON dpt.payment_type_code = ft.type_paiement
    WHERE dd.date_id IS NOT NULL
    """
    
    with engine.connect() as conn:
        conn.execute(text(insert_sql))
        conn.commit()
    
    # Drop temp table
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS fact_transactions_temp"))
        conn.commit()
    
    logger.info(f"  ✓ Loaded fact transactions")

def refresh_aggregates(engine):
    """Refresh Aggregate Tables"""
    logger.info("Refreshing aggregate tables...")
    
    # AGG_DAILY_BY_STATION
    agg_daily_sql = """
    DELETE FROM agg_daily_by_station;
    
    INSERT INTO agg_daily_by_station (date_id, station_id, total_transactions, total_amount, avg_amount, min_amount, max_amount)
    SELECT 
        ft.date_id,
        ft.station_id,
        COUNT(*) as total_transactions,
        SUM(ft.amount_net) as total_amount,
        AVG(ft.amount_net) as avg_amount,
        MIN(ft.amount_net) as min_amount,
        MAX(ft.amount_net) as max_amount
    FROM fact_transactions ft
    GROUP BY ft.date_id, ft.station_id;
    """
    
    with engine.connect() as conn:
        conn.execute(text(agg_daily_sql))
        conn.commit()
    logger.info("  ✓ AGG_DAILY_BY_STATION refreshed")
    
    # AGG_HOURLY_TRAFFIC
    agg_hourly_sql = """
    DELETE FROM agg_hourly_traffic;
    
    INSERT INTO agg_hourly_traffic (date_id, hour, total_vehicles, total_revenue, avg_revenue)
    SELECT 
        ft.date_id,
        EXTRACT(HOUR FROM ft.load_timestamp)::INTEGER as hour,
        COUNT(*) as total_vehicles,
        SUM(ft.amount_net) as total_revenue,
        AVG(ft.amount_net) as avg_revenue
    FROM fact_transactions ft
    GROUP BY ft.date_id, EXTRACT(HOUR FROM ft.load_timestamp);
    """
    
    with engine.connect() as conn:
        conn.execute(text(agg_hourly_sql))
        conn.commit()
    logger.info("  ✓ AGG_HOURLY_TRAFFIC refreshed")

# ====================== MAIN ETL ORCHESTRATION ======================

def run_etl_pipeline():
    """Execute complete ETL pipeline"""
    start_time = datetime.now()
    logger.info("=" * 60)
    logger.info("STARTING DATA WAREHOUSE ETL PIPELINE - TollXpress")
    logger.info("=" * 60)
    
    try:
        # Step 1: Create all tables
        logger.info("\n[STEP 1] Creating Dimension & Fact Tables...")
        create_dim_date_table(engine)
        create_dim_station_table(engine)
        create_dim_vehicle_class_table(engine)
        create_dim_payment_type_table(engine)
        create_fact_transactions_table(engine)
        create_agg_daily_by_station_table(engine)
        create_agg_hourly_traffic_table(engine)
        
        # Step 2: Extract & Transform
        logger.info("\n[STEP 2] Extracting & Transforming Data...")
        df = extract(CSV_FILE)
        df_clean = transform(df)
        
        # Step 3: Load Dimensions
        logger.info("\n[STEP 3] Loading Dimension Tables...")
        load_dimensions(df_clean, engine)
        
        # Step 4: Load Date Dimension
        logger.info("\n[STEP 4] Populating Date Dimension...")
        load_date_dimension(engine)
        
        # Step 5: Load Fact Table
        logger.info("\n[STEP 5] Loading Fact Table...")
        load_fact_table(df_clean, engine)
        
        # Step 6: Refresh Aggregates
        logger.info("\n[STEP 6] Refreshing Aggregate Tables...")
        refresh_aggregates(engine)
        
        duration = datetime.now() - start_time
        logger.info("\n" + "=" * 60)
        logger.info(f"✓ ETL PIPELINE COMPLETED SUCCESSFULLY in {duration.total_seconds():.2f}s")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"✗ ETL FAILED: {str(e)}")
        raise

if __name__ == "__main__":
    run_etl_pipeline()
