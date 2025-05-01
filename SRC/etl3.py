import pandas as pd
import numpy as np
import openpyxl
import io
import os
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

def upload_all_csv_from_folder_to_minio(folder_path, bucket_name):
    """
    Parcourt tous les fichiers CSV dans un dossier local et les envoie dans un bucket MinIO.
    """
    client = get_minio_client()

    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "rb") as file_data:
                file_bytes = file_data.read()
                client.put_object(
                    bucket_name,
                    filename,
                    data=io.BytesIO(file_bytes),
                    length=len(file_bytes),
                    content_type="text/csv",
                )
                print(f"✅ {filename} uploadé dans le bucket MinIO '{bucket_name}'")

# def process_election_2017(file_path):
#     client = get_minio_client()
#     data = client.get_object("datalake", file_path)
#     df = pd.read_excel("data/raw/Presidentielle2017.xlsx", engine='openpyxl', header=0)
#     df.rename(columns={'Code du département': 'code_region'}, inplace=True)
#     # Nettoyage des valeurs non valides (remplacement de NaN par une valeur par défaut, par exemple -1)
#     df['code_region'] = pd.to_numeric(df['code_region'], errors='coerce')  # Force les valeurs invalides à NaN
    
#     # Remplacer les NaN par une valeur spécifique (par exemple -1)
#     df['code_region'].fillna(-1, inplace=True)
    
#     # Convertir la colonne en entier
#     df['code_region'] = df['code_region'].astype('Int64')  # 'Int64' permet de gérer les valeurs manquantes
    
#     # Filtrage sur le département du Gers (code 13)
#     df = df[df['code_region'] == 13]
    
#     df['Département'] = 13
    

#     return df

def process_election(file_path):
    """
    Harmonise un fichier CSV de résultats d'élection présidentielle française,
    quelle que soit l'année ou la structure du fichier.
    Filtre sur le département du Gers (code 13).
    """
    client = get_minio_client()
    data = client.get_object("datalake", file_path)
    df = pd.read_csv(data)

    # 1. Uniformisation des noms de colonnes
    df.columns = (
        df.columns
        .str.lower()
        .str.replace('\xa0', ' ', regex=False)
        .str.replace('-', '_')
        .str.replace(' ', '_')
        .str.replace('é', 'e')
        .str.replace('è', 'e')
        .str.replace('ê', 'e')
        .str.replace('à', 'a')
        .str.replace('ç', 'c')
        .str.strip()
    )

    # 2. Mapping exhaustif des variantes de colonnes
    mapping = {
        'code_du_departement': 'code_region',
        'code_departement': 'code_region',
        'code_de_la_commune': 'code_commune',
        'code_du_canton': 'code_canton',
        'code_de_la_region': 'code_region',
        'code_du_b_vote': 'code_b_vote',
        'etat_saisie': 'etat_saisie',
        'libelle_du_departement': 'nom_departement',
        'libelle_departement': 'nom_departement',
        'libelle_du_canton': 'nom_canton',
        'libelle_de_la_commune': 'nom_commune',
        'inscrits': 'inscrits',
        'abstentions': 'abstentions',
        'pourcentage_abstention_inscrit': 'pourcentage_abstentions_inscrits',
        'pourcentage_abstention_inscrits': 'pourcentage_abstentions_inscrits',
        'pourcentage_votant_inscrit': 'pourcentage_votants_inscrits',
        'pourcentage_votants_inscrits': 'pourcentage_votants_inscrits',
        'votants': 'votants',
        'blancs_et_nuls': 'blancs_et_nuls',
        'blancs-et-nuls': 'blancs_et_nuls',
        'blancs': 'blancs',
        'nuls': 'nuls',
        'exprimés': 'exprimes',
        'exprimes': 'exprimes',
        'pourcentage_blanc_nuls_inscrit': 'pourcentage_blancs_nuls_inscrits',
        'pourcentage_blanc_nuls_votant': 'pourcentage_blancs_nuls_votants',
        'pourcentage_blanc_nuls_inscrits': 'pourcentage_blancs_nuls_inscrits',
        'pourcentage_blanc_nuls_votants': 'pourcentage_blancs_nuls_votants',
        '%_blnuls_ins': 'pourcentage_blancs_nuls_inscrits',
        '%_blnuls_vot': 'pourcentage_blancs_nuls_votants',
        '%_exp_ins': 'pourcentage_exprimes_inscrits',
        '%_exp_vot': 'pourcentage_exprimes_votants',
        '%_nuls_ins': 'pourcentage_nuls_inscrits',
        '%_nuls_vot': 'pourcentage_nuls_votants',
        '%_blancs_ins': 'pourcentage_blancs_inscrits',
        '%_blancs_vot': 'pourcentage_blancs_votants',
        '%_abs_ins': 'pourcentage_abstentions_inscrits',
        '%_vot_ins': 'pourcentage_votants_inscrits',
        'annee': 'annee'
    }
    df.rename(columns=mapping, inplace=True)

    # 3. Harmonisation de la colonne code_region
    region_candidates = ['code_region', 'code_du_departement', 'code_departement']
    for col in region_candidates:
        if col in df.columns:
            df.rename(columns={col: 'code_region'}, inplace=True)
            break
    else:
        raise ValueError(f"'code_region' non trouvée dans {file_path}. Colonnes disponibles : {list(df.columns)}")

    # 4. Conversion des colonnes numériques
    num_cols = [
        'inscrits', 'abstentions', 'votants', 'blancs_et_nuls', 'blancs', 'nuls', 'exprimes',
        'pourcentage_abstentions_inscrits', 'pourcentage_votants_inscrits',
        'pourcentage_blancs_nuls_inscrits', 'pourcentage_blancs_nuls_votants',
        'pourcentage_nuls_inscrits', 'pourcentage_nuls_votants',
        'pourcentage_blancs_inscrits', 'pourcentage_blancs_votants',
        'pourcentage_exprimes_inscrits', 'pourcentage_exprimes_votants'
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df.get(col, np.nan), errors='coerce')

    # 5. Filtrage sur le département du Gers (code 13)
    df['code_region'] = pd.to_numeric(df['code_region'], errors='coerce')
    df = df[df['code_region'] == 13]
    df['departement'] = 13

    # 6. Gestion dynamique des blocs candidats
    candidate_blocks = []
    for i in range(1, 30):
        numbered = []
        for suffix in ['sexe', 'nom', 'prenom', 'voix',
                       'pourcentage_voix_inscrits', 'pourcentage_voix_exprimes', 'n°panneau']:
            # rechercher d'abord les variantes numérotées
            found = False
            for variant in (f'{suffix}_{i}', f'{suffix}{i}', f'{suffix} {i}', f'n_{i}_{suffix}'):
                if variant in df.columns:
                    numbered.append(variant)
                    found = True
            if not found and suffix in df.columns:
                numbered.append(suffix)
        if numbered:
            candidate_blocks.append(numbered)
    if not candidate_blocks:
        base_cols = ['sexe', 'nom', 'prenom', 'voix',
                     'pourcentage_voix_inscrits', 'pourcentage_voix_exprimes', 'n°panneau']
        candidate_blocks = [[col] for col in base_cols if col in df.columns]

    # 7. Transformation en format long (un candidat par ligne)
    id_vars = [c for c in df.columns if not any(c.startswith(pref)
                                                for pref in ['sexe', 'nom', 'prenom', 'voix', 'pourcentage_voix', 'n°panneau'])]
    if len(candidate_blocks) > 1:
        melted = []
        for i, block in enumerate(candidate_blocks, start=1):
            block_cols = [c for c in block if c in df.columns]
            temp = df[id_vars + block_cols].copy()
            temp['candidat_index'] = i
            # Renommer les colonnes du bloc en supprimant suffixe
            rename_cand = {col: col.split('_')[0] for col in block_cols}
            temp.rename(columns=rename_cand, inplace=True)
            # Supprimer colonnes en double issues du renommage
            temp = temp.loc[:, ~temp.columns.duplicated()]
            # Reset index pour éviter conflit d'index
            temp.reset_index(drop=True, inplace=True)
            melted.append(temp)
        # Concaténation finale
        df_long = pd.concat(melted, ignore_index=True)
    else:
        df_long = df.copy()
        df_long['candidat_index'] = df_long.get('candidat_index', 1)

    # 8. Colonnes finales (schéma cible)
    schema = ['annee', 'code_region', 'nom_departement', 'code_canton', 'nom_canton',
              'code_commune', 'nom_commune', 'inscrits', 'abstentions', 'pourcentage_abstentions_inscrits',
              'votants', 'pourcentage_votants_inscrits', 'blancs_et_nuls', 'pourcentage_blancs_nuls_inscrits',
              'pourcentage_blancs_nuls_votants', 'blancs', 'pourcentage_blancs_inscrits', 'pourcentage_blancs_votants',
              'nuls', 'pourcentage_nuls_inscrits', 'pourcentage_nuls_votants', 'exprimes',
              'pourcentage_exprimes_inscrits', 'pourcentage_exprimes_votants',
              'candidat_index', 'n°panneau', 'sexe', 'nom', 'prenom', 'voix',
              'pourcentage_voix_inscrits', 'pourcentage_voix_exprimes', 'departement']
    # S'assurer de la présence de toutes les colonnes
    for col in schema:
        df_long[col] = df_long.get(col, np.nan)
    df_long = df_long[schema]

    return df_long





def process_police(file_name):
    client = get_minio_client()
    data = client.get_object("datalake", file_name)
    df = pd.read_csv(data, sep=";", encoding="utf-8")  # ✅ Spécifie le séparateur ici
    # print("Colonnes disponibles après chargement :", df.columns)  # Debugging
    df.columns = df.columns.str.replace('\xa0', ' ')  # espace insécable
    df.columns = df.columns.str.strip()  # supprime les espaces autour
    return df[df['code_departement'].astype(str).str.startswith('13')]

def process_chomage(file_name):
    client = get_minio_client()
    data = client.get_object("datalake", file_name)
    df = pd.read_csv(data, sep=";", encoding="utf-8")  # ✅ Spécifie le séparateur ici
    # print("Colonnes disponibles après chargement :", df.columns)  # Debugging
    df.columns = df.columns.str.replace('\xa0', ' ')  # espace insécable
    df.columns = df.columns.str.strip()  # supprime les espaces autour  # ✅ Filtrage de la région
    return df[df['departement'].astype(str).str.startswith('Bouches-du-Rhone')]


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
    client = get_minio_client()

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

# fonction pour rendre le code scalable si les fichier son charger sans traitement et transformation en dataframe
# sa le transforme en dataframe 
def csv_from_minio_to_dataframe(bucket_name, file_name):
    """
    Récupère un fichier CSV depuis MinIO et le transforme en DataFrame.
    """
    client = get_minio_client()

    # Télécharge le fichier depuis MinIO
    data = client.get_object(bucket_name, file_name)
    
    # Lire les données du fichier dans un DataFrame
    df = pd.read_csv(io.BytesIO(data.read()))
    
    return df

def send_to_postgresql(df, table_name):
    # Obtenir l'engine SQLAlchemy à partir de l'environnement
    engine = get_sqlalchemy_engine()
    
    # Insérer les données dans la table spécifiée dans PostgreSQL
    df.to_sql(table_name, engine, index=False, if_exists='append')  # Vous pouvez utiliser 'append' si vous ne voulez pas écraser les données
    print(f"✅ Données envoyées vers la base de données dans la table {table_name}")


def delete_from_minio(bucket_name, file_name):
    client = get_minio_client()
    client.remove_object(bucket_name, file_name)
    print(f"🗑️ Fichier {file_name} supprimé du bucket {bucket_name}")