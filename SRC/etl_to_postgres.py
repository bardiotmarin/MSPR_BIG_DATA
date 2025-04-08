import os
import pandas as pd
from minio import Minio
from sqlalchemy import create_engine
from dotenv import load_dotenv
from io import BytesIO

# Chargement des variables d'environnement
load_dotenv()

# ----- CONFIGURATION -----
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT").replace('"', '')
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY").replace('"', '')
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY").replace('"', '')
BUCKET_NAME = os.getenv("BUCKET_NAME").replace('"', '')
FILE_KEY = os.getenv("FILE_KEY").replace('"', '')

DB_HOST = os.getenv("DB_HOST").replace('"', '')
DB_PORT = os.getenv("DB_PORT").replace('"', '')
DB_NAME = os.getenv("DB_NAME").replace('"', '')
DB_USER = os.getenv("DB_USER").replace('"', '')
DB_PASSWORD = os.getenv("DB_PASSWORD").replace('"', '')

# ----- ÉTAPE 1 : Connexion à MinIO et téléchargement du fichier -----
def download_csv_from_minio():
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

    if not client.bucket_exists(BUCKET_NAME):
        raise ValueError(f"Le bucket '{BUCKET_NAME}' n'existe pas.")

    obj = client.get_object(BUCKET_NAME, FILE_KEY)
    data = obj.read()
    return BytesIO(data)

# ----- ÉTAPE 2 : Chargement du fichier CSV -----
def load_csv(byte_stream):
    df = pd.read_csv(byte_stream, sep=";")  # Si problème, ajoute encoding="utf-8"
    return df

# ----- ÉTAPE 3 : Envoi dans PostgreSQL -----
def push_to_postgres(df, table_name):
    engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    print(f"✅ Données insérées dans la table '{table_name}' avec succès.")

# ----- MAIN -----
def main():
    try:
        print("📥 Téléchargement depuis MinIO...")
        byte_stream = download_csv_from_minio()

        print("📊 Chargement du CSV...")
        df = load_csv(byte_stream)

        print("🧠 Aperçu des données :")
        print(df.head())

        print("🚀 Insertion dans PostgreSQL...")
        push_to_postgres(df, "submissions")

    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    main()
