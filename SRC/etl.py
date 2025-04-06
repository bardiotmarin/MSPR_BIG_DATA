import pandas as pd
import openpyxl
import io
from sqlalchemy import create_engine
from src.utils import get_sqlalchemy_engine
from minio import Minio # type: ignore

from src.utils import get_minio_client

def convert_excel_to_csv_and_save_to_minio(excel_file_path, file_name, bucket_name):
    df = pd.read_excel(excel_file_path, engine='openpyxl')
    csv_data = io.BytesIO(df.to_csv(index=False).encode())

    client = get_minio_client()
    
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
    
    client.put_object(
        bucket_name,
        file_name,
        data=csv_data,
        length=len(csv_data.getvalue()),  # Correction ici
        content_type='application/csv',
    )
    print(f"Converti {excel_file_path} en CSV et sauvegardé dans MinIO bucket {bucket_name}")
    
def process_election_2017(file_path):
    client = get_minio_client()
    data = client.get_object("datalake", file_path)
    df = pd.read_excel("data/raw/Presidentielle2017.xlsx", engine='openpyxl', header=0)
    
    # Nettoyage des valeurs non valides (remplacement de NaN par une valeur par défaut, par exemple -1)
    df['Code du département'] = pd.to_numeric(df['Code du département'], errors='coerce')  # Force les valeurs invalides à NaN
    
    # Remplacer les NaN par une valeur spécifique (par exemple -1)
    df['Code du département'].fillna(-1, inplace=True)
    
    # Convertir la colonne en entier
    df['Code du département'] = df['Code du département'].astype('Int64')  # 'Int64' permet de gérer les valeurs manquantes
    
    # Filtrage sur le département du Gers (code 32)
    df = df[df['Code du département'] == 32]
    
    df['Département'] = 32
    return df


def process_resultats_niveau_reg(file_path):
    client = get_minio_client()
    data = client.get_object("datalake", file_path)
    df = pd.read_csv(data)
    df = df[df['Code de la région'] == 32]
    return df


def process_police(file_path):
    df = pd.read_csv(file_path, sep=";", encoding="utf-8")  # ✅ Correction du séparateur

    print("Colonnes disponibles après chargement :", df.columns)  # Debugging

    df.columns = df.columns.str.replace('"', '').str.strip()  # ✅ Nettoyage des colonnes
    print("Colonnes après nettoyage :", df.columns)  # Debugging

    df = df[df['Code_region'].astype(str).str.startswith('32')]  # ✅ Filtrage de la région

    return df

def save_to_minio(df, file_name, bucket_name):
    client = Minio(
        "localhost:9000",  # Modifie si ton MinIO est sur un autre host
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )

    # Convertir le DataFrame en CSV et stocker dans un objet fichier-like
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8")
    csv_buffer.seek(0)  # Revenir au début du fichier

    # Envoyer à MinIO
    client.put_object(
        bucket_name,
        file_name,
        data=csv_buffer,  # 🔥 Correction : on envoie un fichier-like
        length=csv_buffer.getbuffer().nbytes,
        content_type="text/csv",
    )

    print(f"✅ Fichier {file_name} sauvegardé dans MinIO bucket {bucket_name}")

def send_to_postgresql(df, table_name):
    # Obtenir l'engine SQLAlchemy à partir de l'environnement
    engine = get_sqlalchemy_engine()
    
    # Insérer les données dans la table spécifiée dans PostgreSQL
    df.to_sql(table_name, engine, index=False, if_exists='replace')  # Vous pouvez utiliser 'append' si vous ne voulez pas écraser les données
    print(f"✅ Données envoyées vers la base de données dans la table {table_name}")


