import os
from dotenv import load_dotenv
from minio import Minio
from sqlalchemy import create_engine

load_dotenv()

def get_minio_client():
    endpoint = os.getenv('MINIO_ENDPOINT')
    host, port = endpoint.split(":")

    client = Minio(
        f"{host}:{port}",
        access_key=os.getenv('MINIO_ACCESS_KEY'),
        secret_key=os.getenv('MINIO_SECRET_KEY'),
        secure=False,
    )
    return client

def get_or_default_minio_client(client=None):
    """
    Retourne un client MinIO par défaut sauf si un client est fourni (utile pour les tests).
    """
    return client or get_minio_client()

def get_sqlalchemy_engine():
    """
    Crée un moteur SQLAlchemy pour PostgreSQL.
    """
    return create_engine("postgresql+psycopg2://user:password@localhost:5433/mspr_warehouse")
