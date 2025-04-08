import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from src.utils import get_minio_client  # Importez votre utilitaire MinIO
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

# Configuration modifiée
sns.set_theme(style="whitegrid")  # Initialisation du thème seaborn
sns.set_palette("husl")
pd.set_option('display.float_format', '{:.2f}'.format)

def load_from_minio(bucket_name, file_name):
    """Charge un fichier CSV depuis MinIO"""
    minio_client = get_minio_client()
    try:
        response = minio_client.get_object(bucket_name, file_name)
        return pd.read_csv(BytesIO(response.data))
    finally:
        response.close()
        response.release_conn()

def combine_datasets(police_df, election_df):
    """Combine les datasets police et élection"""
    # Ici vous devez implémenter la logique de jointure
    # Cela dépend de la structure exacte de vos données
    
    # Exemple simplifié (à adapter)
    combined_df = pd.merge(
        police_df.groupby('departement').agg({'nombre': 'sum'}).reset_index(),
        election_df,
        left_on='departement',
        right_on='Code du département'
    )
    
    return combined_df

def analyze_correlations(df):
    """Calcule et visualise les corrélations"""
    # Sélection des variables (à adapter selon vos colonnes réelles)
    variables = ['nombre', 'taux_abstention', '% Voix/Exp']  # Exemple
    
    # Matrice de corrélation
    corr_matrix = df[variables].corr(method='pearson')
    
    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title("Matrice de corrélation\nDélinquance × Indicateurs Électoraux")
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.show()
    
    # Corrélations détaillées
    for var in variables[1:]:
        r, p = pearsonr(df['nombre'], df[var])
        print(f"\nCorrélation avec {var}:")
        print(f"- Coefficient r = {r:.3f}")
        print(f"- Significativité p = {p:.4f}")

def temporal_analysis(df):
    """Analyse temporelle conjointe"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Axe 1: Délinquance
    ax1.set_xlabel('Année')
    ax1.set_ylabel('Infractions (total)', color='tab:blue')
    line1 = ax1.plot(df['annee'], df['nombre'], 
                    'o-', color='tab:blue', label='Délinquance')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Axe 2: Indicateur électoral
    ax2 = ax1.twinx()
    ax2.set_ylabel('Taux de vote (%)', color='tab:red')
    line2 = ax2.plot(df['annee'], df['% Voix/Exp'], 
                    's--', color='tab:red', label='Vote candidat X')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    # Légende combinée
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title("Évolution comparée : Délinquance et Résultats Électoraux")
    plt.grid(True, alpha=0.3)
    plt.savefig('evolution_comparative.png')
    plt.show()

def main():
    # Chargement des données depuis MinIO
    try:
        # Charge les données police
        police_df = load_from_minio("datalake", "police_stat_processed.csv")
        
        # Charge les données élection
        election_df = load_from_minio("datalake", "election_2017_processed.csv")
        
        # Combine les datasets
        combined_df = combine_datasets(police_df, election_df)
        
        print("\nAperçu des données combinées:")
        print(combined_df.head())
        
        # Analyses
        analyze_correlations(combined_df)
        
        # Si vous avez des données temporelles
        if 'annee' in combined_df.columns:
            temporal_analysis(combined_df)
        
    except Exception as e:
        print(f"\nErreur: {str(e)}")

if __name__ == "__main__":
    main()