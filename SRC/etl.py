import pandas as pd

def process_election_2017(file_path):
    df = pd.read_excel(file_path)
    # Garder uniquement les données du département 32 et transformer en "32"
    df = df[df['Unnamed: 1'] == 'GERS']
    df['Département'] = 32  # Transformer en entier pour normalisation
    return df

def process_resultats_niveau_reg(file_path):
    df = pd.read_excel(file_path)
    # Garder uniquement les données de la région 32
    df = df[df['code_region'] == 32]
    return df

def process_schmit(file_path):
    df = pd.read_csv(file_path)
    # Garder uniquement les données du département 32 (code_region)
    df = df[df['code_region'].astype(str).str.startswith('32')]
    return df

def save_to_minio(df, file_name, bucket_name):
    from src.utils import get_minio_client
    
    client = get_minio_client()
    
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)
    
    csv_data = df.to_csv(index=False).encode()
    
    client.put_object(
        bucket_name,
        file_name,
        data=csv_data,
        length=len(csv_data),
        content_type='application/csv',
    )
