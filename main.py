from src.etl import convert_excel_to_csv_and_save_to_minio, process_election_2017, process_resultats_niveau_reg, process_police, save_to_minio, send_to_postgresql

import os
from dotenv import load_dotenv

load_dotenv() 
def main():
    # Convertir les fichiers Excel en CSV et les sauvegarder dans MinIO
    convert_excel_to_csv_and_save_to_minio("data/raw/Presidentielle2017.xlsx", "election_2017.csv", "datalake")
    convert_excel_to_csv_and_save_to_minio("data/raw/resultats-par-niveau-reg-t1-france-entiere.xlsx", "resultats_2022_niveau_reg.csv", "datalake")
    
    # Traiter les fichiers CSV depuis MinIO
    election_2017_df = process_election_2017("election_2017.csv")
    resultats_niveau_reg_df = process_resultats_niveau_reg("resultats_2022_niveau_reg.csv")
    police_df = process_police("data/raw/donnee-reg-data.gouv-2024-geographie2024-produit-le2025-01-26.csv")
    
    # Sauvegarder les fichiers traités dans MinIO (datalake)
    save_to_minio(election_2017_df, "election_2017_processed.csv", "datalake")
    save_to_minio(resultats_niveau_reg_df, "resultats_2022_niveau_reg_processed.csv", "datalake")
    save_to_minio(police_df, "police_stat_processed.csv", "datalake")

    # Envoi des DataFrames dans PostgreSQL avec des noms explicites
    send_to_postgresql(election_2017_df, 'election_2017')               # 🗳️ élection 2017
    send_to_postgresql(resultats_niveau_reg_df, 'election_2020')        # 🗳️ élection 2020
    send_to_postgresql(police_df, 'statistique_police')                 # 👮‍♀️ statistiques police

    # Supprimer les fichiers sources non traités après traitement
    try:
        os.remove("election_2017.csv")
        os.remove("resultats_2022_niveau_reg.csv")
        os.remove("data/raw/donnee-reg-data.gouv-2024-geographie2024-produit-le2025-01-26.csv")
        print("🗑️ Fichiers sources supprimés après traitement.")
    except FileNotFoundError:
        print("⚠️ Certains fichiers étaient déjà supprimés ou introuvables.")


if __name__ == "__main__":
    main()
