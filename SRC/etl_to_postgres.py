import pandas as pd
from sqlalchemy import create_engine
from io import BytesIO
from minio import Minio
import os

#Configuration MinIO
MINIO_ENDPOINT = "localhost:9000"
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "minioadmin")
BUCKET_NAME = "datalake"

#Configuration PostgreSQL
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5433")
DB_NAME = os.getenv("DB_NAME", "bigdata")

#Connexion à PostgreSQL via SQLAlchemy
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

#Connexion MinIO
minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False
)

def read_csv_from_minio(object_name):
    """Lit un fichier CSV stocké sur MinIO"""
    response = minio_client.get_object(BUCKET_NAME, object_name)
    data = BytesIO(response.read())
    df = pd.read_csv(data, sep=';')  # Car tes fichiers sont séparés par ;
    return df

def push_to_postgres(df, table_name):
    """Insère un DataFrame dans PostgreSQL"""
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    print(f"✅ Données insérées dans la table '{table_name}'")

def main():
    # Fichiers dans MinIO
    fichiers = {
        "police_stat_processed.csv": "police",
        "election_2017_processed.csv": "election"
    }

    for object_name, table_name in fichiers.items():
        print(f"🔄 Lecture de {object_name} depuis MinIO...")
        df = read_csv_from_minio(object_name)
        print(f"📥 {object_name} chargé. Aperçu :\n{df.head()}")

        print(f"💾 Insertion dans PostgreSQL (table: {table_name})...")
        push_to_postgres(df, table_name)

if __name__ == "__main":
    main()
