import pandas as pd
import openpyxl
import io
# from sqlalchemy import create_engine
from src.utils import get_sqlalchemy_engine
from minio import Minio # type: ignore
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, ForeignKey

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
        length=len(csv_data.getvalue()),
        content_type='application/csv',
    )
    print(f"Converti {excel_file_path} en CSV et sauvegardé dans MinIO bucket {bucket_name}")

def process_election_2017(file_path):
    client = get_minio_client()
    bucket = "datalake"

    data = client.get_object(bucket, file_path)
    df = pd.read_csv(data)

    # Nettoyage initial des noms de colonnes
    df.columns = df.columns.str.replace('"', '').str.strip()

    # Renommer les colonnes AVANT d'y accéder
    df.rename(columns={
        'Libellé du département': 'libelle_departement',
        'Code du canton': 'code_canton',
        'Libellé du canton': 'libelle_canton',
        'Inscrits': 'inscrits',
        'Abstentions': 'abstentions',
        '% Abs/Ins': 'pourcentage_absent_inscrits',
        'Votants': 'votants',
        '% Vot/Ins': 'pourcentage_voix_inscrits',
        'Blancs': 'blancs',
        '% Blancs/Ins': 'pourcentage_blancs_inscrits',
        '% Blancs/Vot': 'pourcentage_blancs_votants',
        'Nuls': 'nuls',
        '% Nuls/Ins': 'pourcentage_nuls_inscrits',
        '% Nuls/Vot': 'pourcentage_nuls_votants',
        'Exprimés': 'exprimes',
        '% Exp/Ins': 'pourcentage_voix_exprimes_inscrits',
        '% Exp/Vot': 'pourcentage_voix_exprimes_votants',
        'Code du département': 'code_region'
        'pourcentage_voix_inscrits': 'pourcentage_voix_inscrits_election',
        'N°Panneau': 'numero_panneau',
        'N°Panneau_1': 'numero_panneau_1',
        'N°Panneau_2': 'numero_panneau_2',
        'N°Panneau_3': 'numero_panneau_3',
        'N°Panneau_4': 'numero_panneau_4',
        'N°Panneau_5': 'numero_panneau_5',
        'N°Panneau_6': 'numero_panneau_6',
        'N°Panneau_7': 'numero_panneau_7',
        'N°Panneau_8': 'numero_panneau_8',
        'N°Panneau_9': 'numero_panneau_9',
        'N°Panneau_10': 'numero_panneau_10',

    }, inplace=True)

    # Maintenant que la colonne 'code_region' existe, on peut la traiter
    df['code_region'] = pd.to_numeric(df['code_region'], errors='coerce')
    df['code_region'] = df['code_region'].fillna(-1).astype('Int64')
    
    # Filtrage sur le Gers (code 32)
    df = df[df['code_region'] == 32]
    df['Région'] = 32

    # Suppression du fichier source de MinIO une fois traité
    delete_from_minio(file_path, bucket)

    return df


def process_resultats_niveau_reg(file_path):
    client = get_minio_client()
    data = client.get_object("datalake", file_path)
    df = pd.read_csv(data)

    df.columns = df.columns.str.replace('\xa0', ' ')
    df.columns = df.columns.str.strip()

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
        '% Voix/Exp': 'pourcentage_voix_exprimes',
        'Code de la région': 'code_region'
    }, inplace=True)
    

    df.rename(columns={'Code du département': 'code_region'}, inplace=True)
    
    # Nettoyage des valeurs non valides (remplacement de NaN par une valeur par défaut, par exemple -1)
    df['code_region'] = pd.to_numeric(df['code_region'], errors='coerce')  # Force les valeurs invalides à NaN
    
    # Remplacer les NaN par une valeur spécifique (par exemple -1)
    df['code_region'] = df['code_region'].fillna(-1)

    
    # Convertir la colonne en entier
    df['code_region'] = df['code_region'].astype('Int64')  # 'Int64' permet de gérer les valeurs manquantes
    
    # Filtrage sur la région du Gers (code 32)
    df = df[df['code_region'] == 32]
    
    df['Région'] = 32  # Si nécessaire, ajouter une colonne Région avec la valeur 32
    df.columns = df.columns.str.replace('"', '').str.strip()

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
    df = df[df['code_region'] == 32]
    
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

    df = df[df['Code_region'].astype(str).str.startswith('32')]  # ✅ Filtrage de la région

    # Renommer la colonne pour correspondre au schéma de la base de données
    df = df.rename(columns={'Code_region': 'code_region'})
    

    return df

# def delete_from_minio(bucket_name, file_name):
 
def delete_from_minio(file_name, bucket_name):
    client = Minio(  #Require log to connect to minio
        "localhost:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )
    client = get_minio_client()
    client.remove_object(bucket_name, file_name)
    print(f"🗑️ Fichier {file_name} supprimé du bucket {bucket_name}")

    # client = get_minio_client()

    try:
        client.remove_object(bucket_name, file_name)
        print(f"🗑️ Fichier {file_name} supprimé du bucket {bucket_name}.")
    except Exception as e:
        print(f"⚠️ Erreur lors de la suppression du fichier {file_name} : {e}")


def save_to_minio(df, file_name, bucket_name):
    client = Minio(
        "localhost:9000",  # Modifie si ton MinIO est sur un autre host
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )

    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8")
    csv_buffer.seek(0)  # Revenir au début du fichier

    client.put_object(
        bucket_name,
        file_name,
        data=csv_buffer,  # 🔥 Correction : on envoie un fichier-like
        length=csv_buffer.getbuffer().nbytes,
        content_type="text/csv",
    )

    print(f"✅ Fichier {file_name} sauvegardé dans MinIO bucket {bucket_name}")


def create_database_schema():
    """
    Crée un schéma de base de données simplifié avec seulement deux tables
    """
    engine = get_sqlalchemy_engine()
    metadata = MetaData()

    # Table élections (format large avec toutes les colonnes du fichier)
    election = Table('election', metadata,
        Column('id', Integer, primary_key=True, autoincrement=True),
        Column('code_region', Integer),
        Column('libelle_departement', String(100)),
        Column('code_canton', String(10)),
        Column('libelle_canton', String(100)),
        Column('inscrits', Integer),
        Column('abstentions', Integer),
        Column('pourcentage_absent_inscrits', Float),
        Column('votants', Integer),
        Column('pourcentage_voix_inscrits_election', Float),
        Column('blancs', Integer),
        Column('pourcentage_blancs_inscrits', Float),
        Column('pourcentage_blancs_votants', Float),
        Column('nuls', Integer),
        Column('pourcentage_nuls_inscrits', Float),
        Column('pourcentage_nuls_votants', Float),
        Column('exprimes', Integer),
        Column('pourcentage_voix_exprimes_inscrits', Float),
        Column('pourcentage_voix_exprimes_votants', Float),
        # Colonnes pour les candidats
        Column('numero_panneau', String(10)),
        Column('sexe', String(1)),
        Column('nom', String(100)),
        Column('prenom', String(100)),
        Column('voix', Integer),
        Column('pourcentage_voix_inscrits', Float),
        Column('pourcentage_voix_exprimes', Float),
        # Colonnes pour les candidats additionnels (1-10)
        # Candidat 1
        Column('numero_panneau_1', String(10)),
        Column('sexe_1', String(1)),
        Column('nom_1', String(100)),
        Column('prenom_1', String(100)),
        Column('voix_1', Integer),
        Column('pourcentage_voix_inscrits_1', Float),
        Column('pourcentage_voix_exprimes_1', Float),
        # Candidat 2
        Column('numero_panneau_2', String(10)),
        Column('sexe_2', String(1)),
        Column('nom_2', String(100)),
        Column('prenom_2', String(100)),
        Column('voix_2', Integer),
        Column('pourcentage_voix_inscrits_2', Float),
        Column('pourcentage_voix_exprimes_2', Float),
        # Candidat 3
        Column('numero_panneau_3', String(10)),
        Column('sexe_3', String(1)),
        Column('nom_3', String(100)),
        Column('prenom_3', String(100)),
        Column('voix_3', Integer),
        Column('pourcentage_voix_inscrits_3', Float),
        Column('pourcentage_voix_exprimes_3', Float),
        # Candidat 4
        Column('numero_panneau_4', String(10)),
        Column('sexe_4', String(1)),
        Column('nom_4', String(100)),
        Column('prenom_4', String(100)),
        Column('voix_4', Integer),
        Column('pourcentage_voix_inscrits_4', Float),
        Column('pourcentage_voix_exprimes_4', Float),
        # Candidat 5
        Column('numero_panneau_5', String(10)),
        Column('sexe_5', String(1)),
        Column('nom_5', String(100)),
        Column('prenom_5', String(100)),
        Column('voix_5', Integer),
        Column('pourcentage_voix_inscrits_5', Float),
        Column('pourcentage_voix_exprimes_5', Float),
        # Candidat 6
        Column('numero_panneau_6', String(10)),
        Column('sexe_6', String(1)),
        Column('nom_6', String(100)),
        Column('prenom_6', String(100)),
        Column('voix_6', Integer),
        Column('pourcentage_voix_inscrits_6', Float),
        Column('pourcentage_voix_exprimes_6', Float),
        # Candidat 7
        Column('numero_panneau_7', String(10)),
        Column('sexe_7', String(1)),
        Column('nom_7', String(100)),
        Column('prenom_7', String(100)),
        Column('voix_7', Integer),
        Column('pourcentage_voix_inscrits_7', Float),
        Column('pourcentage_voix_exprimes_7', Float),
        # Candidat 8
        Column('numero_panneau_8', String(10)),
        Column('sexe_8', String(1)),
        Column('nom_8', String(100)),
        Column('prenom_8', String(100)),
        Column('voix_8', Integer),
        Column('pourcentage_voix_inscrits_8', Float),
        Column('pourcentage_voix_exprimes_8', Float),
        # Candidat 9
        Column('numero_panneau_9', String(10)),
        Column('sexe_9', String(1)),
        Column('nom_9', String(100)),
        Column('prenom_9', String(100)),
        Column('voix_9', Integer),
        Column('pourcentage_voix_inscrits_9', Float),
        Column('pourcentage_voix_exprimes_9', Float),
        # Candidat 10
        Column('numero_panneau_10', String(10)),
        Column('sexe_10', String(1)),
        Column('nom_10', String(100)),
        Column('prenom_10', String(100)),
        Column('voix_10', Integer),
        Column('pourcentage_voix_inscrits_10', Float),
        Column('pourcentage_voix_exprimes_10', Float),
        # Métadonnées
        Column('annee', Integer),
        Column('type_election', String(50))
    )
    police = Table('police', metadata,
        Column('id_stat', Integer, primary_key=True, autoincrement=True),
        Column('code_region', Integer),
        Column('annee', Integer),
        Column('indicateur', String(100)),
        Column('unite_de_compte', String(50)),
        Column('nombre', Integer),
        Column('taux_pour_mille', Float),
        Column('insee_pop', Integer),
        Column('insee_pop_millesime', Integer),
        Column('insee_log', Integer),
        Column('insee_log_millesime', Integer)
    )

    # Création de toutes les tables
    metadata.create_all(engine)
    print("✅ Schéma de base de données créé avec succès.")
    return {
        'election': election,
        'police': police
    }

    # return {
    #     'regions': regions,
    #     'departements': departements,
    #     'cantons': cantons,
    #     'elections': elections,
    #     'resultats_electoraux': resultats_electoraux,
    #     'candidats': candidats,
    #     'votes_candidats': votes_candidats,
    #     'statistiques_police': statistiques_police
    # }

def send_to_postgresql(df, table_name):
    """
    Envoie un DataFrame directement dans une table PostgreSQL
    """
    engine = get_sqlalchemy_engine()
    clean_table_name = table_name.replace(" ", "_").lower()
    
    # Copie du DataFrame pour éviter de modifier l'original
    df_copy = df.copy()
    if "pourcentage_voix_inscrits" in df_copy.columns:
        df_copy.rename(columns={"pourcentage_voix_inscrits": "pourcentage_voix_inscrits_election"}, inplace=True)
    # Conversion des pourcentages (virgules en points)
    for col in df_copy.columns:
        if ('pourcentage' in col or 'taux' in col) and df_copy[col].dtype == 'object':
            df_copy[col] = df_copy[col].str.replace(',', '.').astype(float)
    
    # Déterminer la table cible
    target_table = None
    if "election" in clean_table_name:
        target_table = "election"
        # Ajouter des métadonnées
        if "2017" in clean_table_name:
            df_copy['annee'] = 2017
        elif "2022" in clean_table_name:
            df_copy['annee'] = 2022
        df_copy['type_election'] = 'Présidentielle'
    elif "police" in clean_table_name:
        target_table = "police"
    
    if target_table:
        with engine.connect() as connection:
            # Utiliser to_sql avec if_exists='append' pour ajouter à la table existante
            df_copy.to_sql(target_table, connection, if_exists='append', index=False)
            print(f"✅ Données chargées dans la table {target_table}")
    else:
        print(f"⚠️ Impossible de déterminer la table cible pour {clean_table_name}")
