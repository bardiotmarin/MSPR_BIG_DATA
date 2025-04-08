
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Ajoute le dossier parent
from src.utils import get_minio_client
from minio import Minio
from io import BytesIO

# ⚙️ Paramètres MinIO (adaptés à ton Docker Compose)
minio_client = Minio(
    "localhost:9000",
    access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
    secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
    secure=False
)

bucket_name = "datalake"

def download_from_minio(object_name):
    """Télécharge un fichier CSV depuis MinIO et retourne un DataFrame"""
    try:
        response = minio_client.get_object(bucket_name, object_name)
        df = pd.read_csv(BytesIO(response.read()))
        return df
    except Exception as e:
        print(f"Erreur lors du téléchargement de {object_name} : {e}")
        raise

def load_and_preprocess(election_file, police_file):
    """Charge, fusionne et nettoie les données"""
    df_elec = download_from_minio(election_file)
    df_police = download_from_minio(police_file)

    print("\n✅ Aperçu des fichiers chargés")
    print("Élections :\n", df_elec.head())
    print("Police :\n", df_police.head())

    # 🔁 Merge sur la colonne commune 'annee'
    df = pd.merge(df_police, df_elec, on="annee", how="inner")

    print("\n✅ Colonnes après fusion :\n", df.columns)

    # 🧽 Renommer pour simplifier
    df = df.rename(columns={
        'nombre': 'total_infractions',
        '% Abs/Ins': 'taux_abstention',
        '% Voix/Exp': 'vote_candidat_X'
    })

    # 🎯 Garder les colonnes utiles
    colonnes_utiles = ['annee', 'total_infractions', 'taux_abstention', 'vote_candidat_X']
    df = df[[col for col in colonnes_utiles if col in df.columns]]

    # 🧼 Nettoyage
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    print("\n✅ Données prêtes pour corrélation :\n", df)

    return df

def calculate_correlations(df):
    """Calcule et affiche la corrélation entre colonnes"""
    corr = df.corr()
    print("\n✅ Matrice de corrélation :\n", corr)

    # 🔥 Heatmap
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Matrice de Corrélation")
    plt.show()

    # 📈 Corrélation individuelle (Pearson)
    if 'total_infractions' in df.columns and 'vote_candidat_X' in df.columns:
        x = df['total_infractions']
        y = df['vote_candidat_X']
        if len(x.dropna()) >= 2 and len(y.dropna()) >= 2:
            corr_val, p_val = pearsonr(x, y)
            print(f"\n📌 Corrélation Pearson entre infractions et vote candidat : {corr_val:.3f} (p={p_val:.3f})")
        else:
            print("❌ Pas assez de données valides pour calculer la corrélation Pearson.")

def main():
    # 📂 Fichiers présents dans ton bucket "datalake"
    election_file = "election_2017_processed.csv"
    police_file = "police_stat_processed.csv"

    try:
        df = load_and_preprocess(election_file, police_file)
        calculate_correlations(df)
    except Exception as e:
        print("Erreur:", e)

if __name__ == "__main__":
    main()