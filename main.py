from src.etl import (
    convert_excel_to_csv_and_save_to_minio,
    process_election_2017,
    process_resultats_niveau_reg,
    process_police,
    save_to_minio,
    send_to_postgresql,
    delete_from_minio
)
import os
from dotenv import load_dotenv
from src.utils import get_minio_client

load_dotenv()


def main():
    # === Étape 1 : Conversion & upload des fichiers Excel dans MinIO ===
    convert_excel_to_csv_and_save_to_minio(
        "data/raw/Presidentielle2017.xlsx", 
        "election_2017.csv", 
        "datalake"
    )
    convert_excel_to_csv_and_save_to_minio(
        "data/raw/resultats-par-niveau-reg-t1-france-entiere.xlsx", 
        "resultats_2022_niveau_reg.csv", 
        "datalake"
    )

    # === Étape 2 : Traitement des fichiers ===
    election_2017_df = process_election_2017("election_2017.csv")
    resultats_niveau_reg_df = process_resultats_niveau_reg("resultats_2022_niveau_reg.csv")
    police_df = process_police("data/raw/donnee-reg-data.gouv-2024-geographie2024-produit-le2025-01-26.csv")

    # === Étape 3 : Sauvegarde des fichiers traités dans MinIO ===
    save_to_minio(election_2017_df, "election_2017_processed.csv", "datalake")
    save_to_minio(resultats_niveau_reg_df, "resultats_2022_niveau_reg_processed.csv", "datalake")
    save_to_minio(police_df, "police_stat_processed.csv", "datalake")

    # === Étape 4 : Envoi dans PostgreSQL ===
    send_to_postgresql(election_2017_df, 'election_2017_processed.csv')
    send_to_postgresql(resultats_niveau_reg_df, 'resultats_2022_niveau_reg_processed.csv')
    send_to_postgresql(police_df, 'police_stat_processed.csv')

    # === Étape 5 : Suppression des fichiers locaux ===
    try:
        os.remove("election_2017.csv")
        os.remove("resultats_2022_niveau_reg.csv")
        os.remove("data/raw/donnee-reg-data.gouv-2024-geographie2024-produit-le2025-01-26.csv")
        print("🗑️ Fichiers locaux supprimés après traitement.")
    except FileNotFoundError:
        print("⚠️ Certains fichiers étaient déjà supprimés ou introuvables.")

    # === Étape 6 : Suppression des fichiers bruts depuis MinIO ===
    try:
        delete_from_minio("datalake", "election_2017.csv")
        delete_from_minio("datalake", "resultats_2022_niveau_reg.csv")
        print("🧹 Fichiers CSV bruts supprimés de MinIO.")
    except Exception as e:
        print(f"❌ Erreur lors de la suppression depuis MinIO : {e}")


if __name__ == "__main__":
    main()
