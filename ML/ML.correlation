import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

# Configuration modifiée
sns.set_theme(style="whitegrid")  # Initialisation du thème seaborn
sns.set_palette("husl")
pd.set_option('display.float_format', '{:.2f}'.format)

def load_and_preprocess(filepath):
    """Charge et prépare les données combinées"""
    df = pd.read_excel(filepath)
    
    # Nettoyage des colonnes (adaptez selon vos noms réels)
    df = df.rename(columns={
        'nombre': 'total_infractions',
        '% Abs/Ins': 'taux_abstention',
        '% Voix/Exp': 'vote_candidat_X'  # À remplacer par le vrai nom
    })
    
    # Agrégation par année si nécessaire
    if 'Code du département' in df.columns:
        df = df.groupby('annee').agg({
            'total_infractions': 'sum',
            'taux_abstention': 'mean',
            'vote_candidat_X': 'mean'
        }).reset_index()
    
    return df

def analyze_correlations(df):
    """Calcule et visualise les corrélations"""
    # Sélection des variables
    variables = ['total_infractions', 'taux_abstention', 'vote_candidat_X']
    
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
        r, p = pearsonr(df['total_infractions'], df[var])
        print(f"\nCorrélation avec {var}:")
        print(f"- Coefficient r = {r:.3f}")
        print(f"- Significativité p = {p:.4f}")

def temporal_analysis(df):
    """Analyse temporelle conjointe"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Axe 1: Délinquance
    ax1.set_xlabel('Année')
    ax1.set_ylabel('Infractions (total)', color='tab:blue')
    line1 = ax1.plot(df['annee'], df['total_infractions'], 
                    'o-', color='tab:blue', label='Délinquance')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    
    # Axe 2: Indicateur électoral
    ax2 = ax1.twinx()
    ax2.set_ylabel('Taux de vote (%)', color='tab:red')
    line2 = ax2.plot(df['annee'], df['vote_candidat_X'], 
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
    # Chemin vers votre fichier combiné
    DATA_PATH = "C:\\Users\\droui\\Documents\\githubprojetcs\\MSPR_BIG_DATA\\DATA\\raw\\fichierdelinquancepresidence.xlsx"
    
    # Chargement
    try:
        df = load_and_preprocess(DATA_PATH)
        print("\nAperçu des données:")
        print(df.head())
        
        # Analyses
        analyze_correlations(df)
        temporal_analysis(df)
        
    except Exception as e:
        print(f"\nErreur: {str(e)}")

if __name__ == "__main__":
    main()