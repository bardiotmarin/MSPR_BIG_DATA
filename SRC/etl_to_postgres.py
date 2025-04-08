import os
import io
import pandas as pd
from minio import Minio
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Récupération des variables d'environnement
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT").replace('"', '')
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY").replace('"', '')
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY").replace('"', '')
BUCKET_NAME = os.getenv("BUCKET_NAME").replace('"', '')
FILE_KEY = "election_2017_processed.csv"  # Nom exact du fichier sur MinIO

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


def get_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )


def download_csv_from_minio(file_key):
    print("📥 Téléchargement depuis MinIO...")

    client = get_minio_client()

    try:
        data = client.get_object(BUCKET_NAME, file_key)
        csv_bytes = data.read()
        print("✅ Fichier téléchargé avec succès.")
        return io.BytesIO(csv_bytes)
    except Exception as e:
        print(f"❌ Erreur lors du téléchargement : {e}")
        return None


def load_dataframe_from_csv(csv_file):
    try:
        df = pd.read_csv(csv_file, sep=';')  # Important pour séparer correctement les colonnes
        print("✅ Données chargées dans un DataFrame.")
        return df
    except Exception as e:
        print(f"❌ Erreur de lecture du CSV : {e}")
        return None


def push_to_postgres(df, table_name):
    print("📤 Insertion dans PostgreSQL...")

    try:
        engine = create_engine(
            f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        print("✅ Données insérées avec succès dans la table PostgreSQL.")
    except Exception as e:
        print(f"❌ Erreur lors de l'insertion dans PostgreSQL : {e}")


def main():
    file_key = FILE_KEY  # Modifiable si tu veux tester avec d'autres fichiers
    table_name = "election_2017"  # Nom de la table cible dans PostgreSQL

    csv_file = download_csv_from_minio(file_key)
    if csv_file is None:
        return

    df = load_dataframe_from_csv(csv_file)
    if df is None:
        return

    push_to_postgres(df, table_name)


if __name__ == "__main__":
    main()
