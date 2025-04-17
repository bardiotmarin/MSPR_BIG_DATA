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
    data = client.get_object("datalake", file_path)
    df = pd.read_excel("data/raw/Presidentielle2017.xlsx", engine='openpyxl', header=0)
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
    'Sexe': 'sexe',
    'Nom': 'nom',
    'Prénom': 'prenom',
    'Voix': 'voix',
    '% Voix/Ins': 'pourcentage_voix_inscrits',
    '% Voix/Exp': 'pourcentage_voix_exprimes',
     'N°Panneau': 'numero_panneau',
    'sexe': 'sexe',
    'nom': 'nom',
    'prenom': 'prenom',
    'voix': 'voix',
    'pourcentage_voix_inscrits': 'pourcentage_voix_inscrits',
    'pourcentage_voix_exprimes': 'pourcentage_voix_exprimes',
    'N°Panneau.1': 'numero_panneau_1',
    'Sexe.1': 'sexe_1',
    'Nom.1': 'nom_1',
    'Prénom.1': 'prenom_1',
    'Voix.1': 'voix_1',
    '% Voix/Ins.1': 'pourcentage_voix_inscrits_1',
    '% Voix/Exp.1': 'pourcentage_voix_exprimes_1',
    'N°Panneau.2': 'numero_panneau_2',
    'Sexe.2': 'sexe_2',
    'Nom.2': 'nom_2',
    'Prénom.2': 'prenom_2',
    'Voix.2': 'voix_2',
    '% Voix/Ins.2': 'pourcentage_voix_inscrits_2',
    '% Voix/Exp.2': 'pourcentage_voix_exprimes_2',
    'N°Panneau.3': 'numero_panneau_3',
    'Sexe.3': 'sexe_3',
    'Nom.3': 'nom_3',
    'Prénom.3': 'prenom_3',
    'Voix.3': 'voix_3',
    '% Voix/Ins.3': 'pourcentage_voix_inscrits_3',
    '% Voix/Exp.3': 'pourcentage_voix_exprimes_3',
    'N°Panneau.4': 'numero_panneau_4',
    'Sexe.4': 'sexe_4',
    'Nom.4': 'nom_4',
    'Prénom.4': 'prenom_4',
    'Voix.4': 'voix_4',
    '% Voix/Ins.4': 'pourcentage_voix_inscrits_4',
    '% Voix/Exp.4': 'pourcentage_voix_exprimes_4',
    'N°Panneau.5': 'numero_panneau_5',
    'Sexe.5': 'sexe_5',
    'Nom.5': 'nom_5',
    'Prénom.5': 'prenom_5',
    'Voix.5': 'voix_5',
    '% Voix/Ins.5': 'pourcentage_voix_inscrits_5',
    '% Voix/Exp.5': 'pourcentage_voix_exprimes_5',
    'N°Panneau.6': 'numero_panneau_6',
    'Sexe.6': 'sexe_6',
    'Nom.6': 'nom_6',
    'Prénom.6': 'prenom_6',
    'Voix.6': 'voix_6',
    '% Voix/Ins.6': 'pourcentage_voix_inscrits_6',
    '% Voix/Exp.6': 'pourcentage_voix_exprimes_6',
    'N°Panneau.7': 'numero_panneau_7',
    'Sexe.7': 'sexe_7',
    'Nom.7': 'nom_7',
    'Prénom.7': 'prenom_7',
    'Voix.7': 'voix_7',
    '% Voix/Ins.7': 'pourcentage_voix_inscrits_7',
    '% Voix/Exp.7': 'pourcentage_voix_exprimes_7',
    'N°Panneau.8': 'numero_panneau_8',
    'Sexe.8': 'sexe_8',
    'Nom.8': 'nom_8',
    'Prénom.8': 'prenom_8',
    'Voix.8': 'voix_8',
    '% Voix/Ins.8': 'pourcentage_voix_inscrits_8',
    '% Voix/Exp.8': 'pourcentage_voix_exprimes_8',
    'N°Panneau.9': 'numero_panneau_9',
    'Sexe.9': 'sexe_9',
    'Nom.9': 'nom_9',
    'Prénom.9': 'prenom_9',
    'Voix.9': 'voix_9',
    '% Voix/Ins.9': 'pourcentage_voix_inscrits_9',
    '% Voix/Exp.9': 'pourcentage_voix_exprimes_9',
    'N°Panneau.10': 'numero_panneau_10',
    'Sexe.10': 'sexe_10',
    'Nom.10': 'nom_10',
    'Prénom.10': 'prenom_10',
    'Voix.10': 'voix_10',
    '% Voix/Ins.10': 'pourcentage_voix_inscrits_10',
    '% Voix/Exp.10': 'pourcentage_voix_exprimes_10', 
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
    Crée le schéma de base de données avec toutes les tables nécessaires
    """
    engine = get_sqlalchemy_engine()
    metadata = MetaData()

    # Définition des tables
    regions = Table('regions', metadata,
        Column('code_region', Integer, primary_key=True),
        Column('libelle_region', String(100))
    )

    departements = Table('departements', metadata,
        Column('code_departement', Integer, primary_key=True),
        Column('libelle_departement', String(100)),
        Column('code_region', Integer, ForeignKey('regions.code_region'))
    )

    cantons = Table('cantons', metadata,
        Column('id_canton', Integer, primary_key=True, autoincrement=True),
        Column('code_canton', String(10)),
        Column('libelle_canton', String(100)),
        Column('code_departement', Integer, ForeignKey('departements.code_departement'))
    )

    elections = Table('elections', metadata,
        Column('id_election', Integer, primary_key=True, autoincrement=True),
        Column('annee', Integer),
        Column('type_election', String(50))
    )

    resultats_electoraux = Table('resultats_electoraux', metadata,
        Column('id_resultat', Integer, primary_key=True, autoincrement=True),
        Column('id_election', Integer, ForeignKey('elections.id_election')),
        Column('code_region', Integer, ForeignKey('regions.code_region')),
        Column('code_departement', Integer, ForeignKey('departements.code_departement')),
        Column('id_canton', Integer, ForeignKey('cantons.id_canton')),
        Column('inscrits', Integer),
        Column('abstentions', Integer),
        Column('votants', Integer),
        Column('blancs', Integer),
        Column('nuls', Integer),
        Column('exprimes', Integer)
    )

    candidats = Table('candidats', metadata,
        Column('id_candidat', Integer, primary_key=True, autoincrement=True),
        Column('nom', String(100)),
        Column('prenom', String(100)),
        Column('sexe', String(1))
    )

    votes_candidats = Table('votes_candidats', metadata,
        Column('id_vote', Integer, primary_key=True, autoincrement=True),
        Column('id_resultat', Integer, ForeignKey('resultats_electoraux.id_resultat')),
        Column('id_candidat', Integer, ForeignKey('candidats.id_candidat')),
        Column('voix', Integer),
        Column('pourcentage_voix_inscrits', Float),
        Column('pourcentage_voix_exprimes', Float)
    )

    statistiques_police = Table('statistiques_police', metadata,
        Column('id_stat', Integer, primary_key=True, autoincrement=True),
        Column('code_region', Integer, ForeignKey('regions.code_region')),
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
        'regions': regions,
        'departements': departements,
        'cantons': cantons,
        'elections': elections,
        'resultats_electoraux': resultats_electoraux,
        'candidats': candidats,
        'votes_candidats': votes_candidats,
        'statistiques_police': statistiques_police
    }


def send_to_postgresql(df, table_name):
    engine = get_sqlalchemy_engine()
    clean_table_name = table_name.replace(" ", "_").lower()

    # if "election_2017" in clean_table_name:
    #     # Transformations for 2017 elections
    #     election_2017_df = df.copy()
        
    #     # Rename column to match schema
    #     if 'code_region' in election_2017_df.columns:
    #         election_2017_df = election_2017_df.rename(columns={'code_region': 'code_region'})
        
    #     # Check data type before applying string operations
    #     if 'pourcentage_voix_inscrits' in election_2017_df.columns:
    #         # Then access the column
    #         if election_2017_df['pourcentage_voix_inscrits'].dtype == 'object':
    #             election_2017_df['pourcentage_voix_inscrits'] = election_2017_df['pourcentage_voix_inscrits'].str.replace(',', '.').astype(float)

            
    if "police_stat" in clean_table_name:
        # Transformations for police statistics
        police_df = df.copy()
        
        # Rename column to match schema
        if 'Code_region' in police_df.columns:
            police_df = police_df.rename(columns={'Code_region': 'code_region'})
        
        # Check data type before applying string operations
        if 'taux_pour_mille' in police_df.columns:
            if police_df['taux_pour_mille'].dtype == 'object':  # If it's a string
                police_df['taux_pour_mille'] = police_df['taux_pour_mille'].str.replace(',', '.').astype(float)
            # If it's already numeric, no conversion needed
        
        with engine.connect() as connection:
            police_df.to_sql('statistiques_police', connection, if_exists='append', index=False)
    
    print(f"✅ Données transformées et chargées dans le schéma normalisé pour {clean_table_name}")

