import os
from dotenv import load_dotenv


load_dotenv()

def get_minio_client():
    from minio import Minio
    import os

    endpoint = os.getenv('MINIO_ENDPOINT')
    host, port = endpoint.split(":")  # Séparation hôte/port

    client = Minio(
        f"{host}:{port}",
        access_key=os.getenv('MINIO_ACCESS_KEY'),
        secret_key=os.getenv('MINIO_SECRET_KEY'),
        secure=False,
    )
    return client


def get_sqlalchemy_engine():
    from sqlalchemy import create_engine
    
    return create_engine("postgresql+psycopg2://user:password@localhost:5433/mspr_warehouse")
