import os
from dotenv import load_dotenv

load_dotenv()

def get_minio_client():
    from minio import Minio
    import os

    endpoint = os.getenv('MINIO_ENDPOINT')
    host, port = endpoint.split(":")  # Séparation hôte/port

    client = Minio(
        f"{host}:{port}",  # Assurez-vous que le format est correct
        access_key=os.getenv('MINIO_ACCESS_KEY'),
        secret_key=os.getenv('MINIO_SECRET_KEY'),
        secure=False,
    )
    return client


def get_sqlalchemy_engine():
    from sqlalchemy import create_engine
    
    connection_string = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    return create_engine(connection_string)
