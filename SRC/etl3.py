import pandas as pd
import openpyxl
import io
from sqlalchemy import create_engine
from src.utils import get_sqlalchemy_engine
from minio import Minio # type: ignore

from src.utils2 import get_minio_client

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
        length=len(csv_data.getvalue()),    
        content_type='application/csv',
    )
    print(f"Converti {excel_file_path} en CSV et sauvegardé dans MinIO bucket {bucket_name}")
    
def process_election_2017(file_path):
    client = get_minio_client()
    data = client.get_object("datalake", file_path)
    df = pd.read_excel("data/raw/Presidentielle2017.xlsx", engine='openpyxl', header=0)
    df.rename(columns={'Code du département': 'code_region'}, inplace=True)
    # Nettoyage des valeurs non valides (remplacement de NaN par une valeur par défaut, par exemple -1)
    df['code_region'] = pd.to_numeric(df['code_region'], errors='coerce')  # Force les valeurs invalides à NaN
    
    # Remplacer les NaN par une valeur spécifique (par exemple -1)
    df['code_region'].fillna(-1, inplace=True)
    
    # Convertir la colonne en entier
    df['code_region'] = df['code_region'].astype('Int64')  # 'Int64' permet de gérer les valeurs manquantes
    
    # Filtrage sur le département du Gers (code 32)
    df = df[df['code_region'] == 13]
    
    df['Département'] = 13
    

    return df

 
def process_resultats_niveau_reg(file_path):
    client = get_minio_client()
    data = client.get_object("datalake", file_path)
    df = pd.read_csv(data)
    df.columns = df.columns.str.replace('\xa0', ' ')  # espace insécable
    df.columns = df.columns.str.strip()  # supprime les espaces autour

    df.rename(columns={
    'Libellé de la région': 'nom_region',
    'Etat saisie': 'etat_saisie',
    'Inscrits': 'inscrits',
    'Abstentions': 'abstentions',
    '% Abs/Ins': 'pourcentage_abstentions_inscrits',
    'Votants': 'votants',
    '% Vot/Ins': 'pourcentage_votants_inscrits',
    'Blancs': 'blancs',
    '% Blancs/Ins': 'pourcentage_blancs_inscrits',
    '% Blancs/Vot': 'pourcentage_blancs_votants',
    'Nuls': 'nuls',
    '% Nuls/Ins': 'pourcentage_nuls_inscrits',
    '% Nuls/Vot': 'pourcentage_nuls_votants',
    'Exprimés': 'exprimes',
    '% Exp/Ins': 'pourcentage_exprimes_inscrits',
    '% Exp/Vot': 'pourcentage_exprimes_votants',
    'Sexe': 'sexe',
    'Nom': 'nom',
    'Prénom': 'prenom',
    'Voix': 'voix',
    '% Voix/Ins': 'pourcentage_voix_inscrits',
    '% Voix/Exp': 'pourcentage_voix_exprimes'
    }, inplace=True)
    df.rename(columns={'% Voix/Voix/Ins': 'pourcentage_voix_inscrits'}, inplace=True)
    df.rename(columns={'% Voix/Exp': 'pourcentage_voix_inscrits'}, inplace=True)
    df.rename(columns={'Code de la région': 'code_region'}, inplace=True)
    # Filtrer pour ne garder que la région 32 (Hauts-de-France)
    df = df[df['code_region'] == 13]
    
    # Créer une liste pour stocker les nouvelles lignes
    new_rows = []
    
    # Pour chaque ligne du DataFrame original
    for _, row in df.iterrows():
        # Extraire les informations communes à tous les candidats
        common_data = row.iloc[:17].to_dict()  # Colonnes jusqu'à "% Exp/Vot"
        
        # Parcourir les candidats (regroupés par 6 colonnes: Sexe, Nom, Prénom, Voix, % Voix/Ins, % Voix/Exp)
        for i in range(17, len(row), 6):
            if i+5 < len(row) and pd.notna(row[i]) and pd.notna(row[i+1]):  # Vérifier que les données existent
                candidate_data = {
                    **common_data,
                    'Sexe': row[i],
                    'Nom': row[i+1],
                    'Prénom': row[i+2],
                    'Voix': row[i+3],
                    '% Voix/Ins': row[i+4],
                    '% Voix/Exp': row[i+5]
                }
                new_rows.append(candidate_data)
    df.columns = df.columns.str.replace('"', '').str.strip()

    
    # Créer un nouveau DataFrame à partir des lignes transformées
    result_df = pd.DataFrame(new_rows)
    
    # Supprimer toutes les colonnes qui commencent par "Unnamed:"
    df = df.rename(columns={'Code de la région': 'code_region'})
    unnamed_columns = [col for col in result_df.columns if col.startswith('Unnamed:')]
    if unnamed_columns:
        result_df = result_df.drop(columns=unnamed_columns)
        print(f"🗑️ {len(unnamed_columns)} colonnes 'Unnamed' supprimées du DataFrame.")
    
    return result_df


def process_police(file_path):
    df = pd.read_csv(file_path, sep=";", encoding="utf-8")  # ✅ Correction du séparateur

    print("Colonnes disponibles après chargement :", df.columns)  # Debugging

    df.columns = df.columns.str.replace('"', '').str.strip()  # ✅ Nettoyage des colonnes
    print("Colonnes après nettoyage :", df.columns)  # Debugging
    df.rename(columns={'Code_region': 'code_region'}, inplace=True)

    df = df[df['code_region'].astype(str).str.startswith('13')]  # ✅ Filtrage de la région

    return df

def process_election_results(df):
    # Supprimer les colonnes inutiles (Unnamed: 23 à Unnamed: 88)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Transformer les résultats pour chaque personne
    rows = []
    for _, row in df.iterrows():
        people_data = row[["Sexe", "Nom", "Prénom", "Voix", "% Voix/Ins", "% Voix/Exp"]]
        rows.append(people_data)

    # Créer un DataFrame à partir des lignes transformées
    processed_df = pd.DataFrame(rows, columns=["Sexe", "Nom", "Prénom", "Voix", "% Voix/Ins", "% Voix/Exp"])

    # Retourner le DataFrame nettoyé et transformé
    return processed_df


def save_to_minio(df, file_name, bucket_name):
    client = Minio(
        "localhost:9000",  # Modifie si ton MinIO est sur un autre host
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )

    # Convertir le DataFrame en CSV et stocker dans un objet fichier-like
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False, sep=',', encoding="utf-8")  # Ajout du séparateur explicite ',' et suppression de l'index
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


def delete_from_minio(bucket_name, file_name):
    client = get_minio_client()
    client.remove_object(bucket_name, file_name)
    print(f"🗑️ Fichier {file_name} supprimé du bucket {bucket_name}")