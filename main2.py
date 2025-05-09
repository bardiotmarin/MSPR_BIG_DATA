import os
import re
from dotenv import load_dotenv
from src.etl3 import process_vote_comportement, push_or_flush_postgresql,process_chomage,csv_from_minio_to_dataframe,convert_excel_to_csv_and_save_to_minio,upload_all_csv_from_folder_to_minio,process_election_results, process_election, process_police, save_to_minio, send_to_postgresql


load_dotenv() 

def main():
    # convert_excel_to_csv_and_save_to_minio("data/raw/Presidentielle2017.xlsx", "election_2017.csv", "datalake")
    # convert_excel_to_csv_and_save_to_minio("data/raw/resultats-par-niveau-reg-t1-france-entiere.xlsx", "election_2022.csv", "datalake")
    # convert_excel_to_csv_and_save_to_minio("data/raw/Presidentielle-2017-T1.xlsx", "presidentielle_2017_tour_1.csv", "datalake")
    # convert_excel_to_csv_and_save_to_minio("data/raw/donnee-reg-data.gouv-2024-geographie2024-produit-le2025-01-26.xlsx", "police_stat.csv", "datalake")
    upload_all_csv_from_folder_to_minio("data/raw", "datalake")

    # Traiter les fichiers CSV depuis MinIO
    # election_2017_df = process_election_2017("election_2017.csv")
    # resultats_2022_df = process_election("election_2022.csv")
    resultats_2002_T1_df = process_election("Presidentielle-2002-T1.csv")
    resultats_2002_T2_df = process_election("Presidentielle-2002-T2.csv")
    resultats_2007_T1_df = process_election("Presidentielle-2007-T1.csv")
    resultats_2007_T2_df = process_election("Presidentielle-2007-T2.csv")
    resultats_2012_T1_df = process_election("Presidentielle-2012-T1.csv")
    resultats_2012_T2_df = process_election("Presidentielle-2012-T2.csv")
    resultats_2017_T1_df = process_election("Presidentielle-2017-T1.csv")
    resultats_2017_T2_df = process_election("Presidentielle-2017-T2.csv")
    resultats_2022_T1_df = process_election("Presidentielle-2022-T1.csv")
    resultats_2022_T2_df = process_election("Presidentielle-2022-T2.csv")
    pauvrete_df = csv_from_minio_to_dataframe("datalake", "pauvrete.csv")
    police_df = process_police("statistiques_police.csv")
    chomage_df = process_chomage("chomage.csv")
    comportements_vote_df = csv_from_minio_to_dataframe("datalake", "comportements_vote.csv")
    comportement_votes_metier_df = process_vote_comportement("comportement_votes_metier.csv")
    comportement_votes_lieu_df = process_vote_comportement("comportement_votes_metier.csv")
   

 # Sauvegarder les fichiers traités dans MinIO (datalake)
    # ELECTION 
    save_to_minio(resultats_2002_T1_df , "election_2002_tour_1_processed.csv", "datalake")
    save_to_minio(resultats_2002_T2_df , "election_2002_tour_2_processed.csv", "datalake")
    save_to_minio(resultats_2007_T1_df , "election_2007_tour_1_processed.csv", "datalake")
    save_to_minio(resultats_2007_T2_df , "election_2007_tour_2_processed.csv", "datalake")
    save_to_minio(resultats_2012_T1_df , "election_2012_tour_1_processed.csv", "datalake")
    save_to_minio(resultats_2012_T2_df , "election_2012_tour_2_processed.csv", "datalake")
    save_to_minio(resultats_2017_T1_df , "election_2017_tour_1_processed.csv", "datalake")
    save_to_minio(resultats_2017_T2_df , "election_2017_tour_2_processed.csv", "datalake")
    save_to_minio(resultats_2022_T1_df , "election_2022_tour_1_processed.csv", "datalake")
    save_to_minio(resultats_2022_T2_df , "election_2022_tour_2_processed.csv", "datalake")
    
    save_to_minio(comportements_vote_df , "comportements_vote_processed.csv", "datalake")
    save_to_minio(comportement_votes_metier_df , "comportements_vote_metiers_processed.csv", "datalake")
    save_to_minio(comportement_votes_lieu_df , "comportements_vote_lieux_processed.csv", "datalake")
    save_to_minio(pauvrete_df , "pauvrete_processed.csv", "datalake")
    save_to_minio(chomage_df , "chomage_processed.csv", "datalake")
    save_to_minio(police_df , "statistique_police_processed.csv", "datalake")
    # save_to_minio(resultats_2022_df , "resultats_2022_niveau_reg.csv", "datalake")
    # save_to_minio(police_df, "police_stat_processed.csv", "datalake")

# Envoi des DataFrames dans PostgreSQL avec des noms explicites
    # ELECTION 
    send_to_postgresql(resultats_2002_T1_df , 'election_tour_1')    
    send_to_postgresql(resultats_2007_T1_df , 'election_tour_1')
    send_to_postgresql(resultats_2012_T1_df , 'election_tour_1') 
    send_to_postgresql(resultats_2017_T1_df , 'election_tour_1') 
    send_to_postgresql(resultats_2022_T1_df , 'election_tour_1')             # 🗳️ élection 2017
    send_to_postgresql(resultats_2002_T2_df , 'election_tour_2')
    send_to_postgresql(resultats_2007_T2_df , 'election_tour_2')
    send_to_postgresql(resultats_2012_T2_df , 'election_tour_2')
    send_to_postgresql(resultats_2017_T2_df , 'election_tour_2')
    send_to_postgresql(resultats_2022_T2_df , 'election_tour_2')
    # COMPORTEMENT DES VOTES
    push_or_flush_postgresql(comportements_vote_df , "comportements_vote_france")
    push_or_flush_postgresql(comportement_votes_metier_df , "comportements_vote_metiers_france")
    push_or_flush_postgresql(comportement_votes_lieu_df , "comportements_vote_lieux_france")


    # STATISTIQUE CONNEX 
    push_or_flush_postgresql(pauvrete_df , "pauvrete_france")
    push_or_flush_postgresql(chomage_df , "chomage_france")
    push_or_flush_postgresql(police_df , "police_et_gendarmerie_statistique_france")




    # send_to_postgresql(resultats_2022_df, 'election_tour_1')        # 🗳️ élection 2020
    # send_to_postgresql(police_df, 'statistiques_police')                 # 👮‍♀️ statistiques police

    # Supprimer les fichiers sources non traités après traitement
    # try:
    #     os.remove("election_2017.csv")
    #     os.remove("presidentielle_2017_tour_1.csv")
    #     os.remove("presidentielle_2017_tour_2.csv")
    #     os.remove("resultats_2022_niveau_reg.csv")
    #     os.remove("data/raw/donnee-reg-data.gouv-2024-geographie2024-produit-le2025-01-26.csv")
    #     print("🗑️ Fichiers sources supprimés après traitement.")
    # except FileNotFoundError:
    #     print("⚠️ Certains fichiers étaient déjà supprimés ou introuvables.")


if __name__ == "__main__":
    main()