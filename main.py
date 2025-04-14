from src.etl import convert_excel_to_csv_and_save_to_minio, process_election_2017, process_resultats_niveau_reg, process_police, save_to_minio, send_to_postgresql , delete_from_minio, create_database_schema
import os
from dotenv import load_dotenv
from src.utils import get_minio_client

load_dotenv()


def main():
    # Créer le schéma de base de données
    create_database_schema()
    
    # Le reste de votre code existant...
    # Convertir les fichiers Excel en CSV et les sauvegarder dans MinIO
    convert_excel_to_csv_and_save_to_minio("data/raw/Presidentielle2017.xlsx", "election_2017.csv", "datalake")
    convert_excel_to_csv_and_save_to_minio("data/raw/resultats-par-niveau-reg-t1-france-entiere.xlsx", "election_2022.csv", "datalake")
    
    # Traiter les fichiers CSV depuis MinIO
    election_2017_df = process_election_2017("election_2017.csv")
    election_2022_df = process_resultats_niveau_reg("election_2022.csv")
    police_df = process_police("data/raw/donnee-reg-data.gouv-2024-geographie2024-produit-le2025-01-26.csv")

    # === Étape 3 : Sauvegarde des fichiers traités dans MinIO ===
    save_to_minio(election_2017_df, "election_2017_processed.csv", "datalake")
    save_to_minio(election_2022_df, "election_2022_processed.csv", "datalake")
    save_to_minio(police_df, "police_stat_processed.csv", "datalake")
    
    # Supprimer les fichiers sources non traités après traitement depuis MinIO
    delete_from_minio("election_2017.csv", "datalake")
    delete_from_minio("election_2022.csv", "datalake")

    # Envoi des DataFrames dans PostgreSQL avec transformation
    send_to_postgresql(election_2017_df, 'election_2017_processed')
    send_to_postgresql(election_2022_df, 'election_2022_df_processed')
    send_to_postgresql(police_df, 'police_stat_processed')

    # # === Étape 6 : Suppression des fichiers bruts depuis MinIO ===
    # try:
    #     delete_from_minio("datalake", "election_2017.csv")
    #     delete_from_minio("datalake", "resultats_2022_niveau_reg.csv")
    #     print("🧹 Fichiers CSV bruts supprimés de MinIO.")
    # except Exception as e:
    #     print(f"❌ Erreur lors de la suppression depuis MinIO : {e}")


if __name__ == "__main__":
    main()
