from src.etl import process_election_2017, process_resultats_niveau_reg, process_police, save_to_minio

def main():
    # Traiter les fichiers Excel et CSV bruts dans le dossier "data/raw/"
    
    election_2017_df = process_election_2017("data/raw/Presidentielle_2017_Resultats_Tour_1.xlsx")
    resultats_niveau_reg_df = process_resultats_niveau_reg("data/raw/resultats-par-niveau-reg-t1-france-entiere.xlsx")
    police_df = process_police("data/raw/donnee-reg-data.gouv-2024-geographie2024-produit-le2025-01-26.csv")
    
    # Sauvegarder les fichiers traités dans MinIO (datalake)
    
    save_to_minio(election_2017_df, "election_2017_processed.csv", "election-data")
    save_to_minio(resultats_niveau_reg_df, "resultats_2022_niveau_reg_processed.csv", "election-data")
    save_to_minio(police_df, "police_stat_processed.csv", "election-data")
    
if __name__ == "__main__":
    main()
