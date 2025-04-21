import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings
import os
import sys
from pathlib import Path

# Configuration du path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import des utils
from src.utils import get_sqlalchemy_engine

# Configuration du style
plt.style.use('ggplot')
sns.set_palette("husl")
warnings.filterwarnings("ignore")

def load_data():
    """Charge les données depuis PostgreSQL avec gestion robuste des colonnes"""
    engine = get_sqlalchemy_engine()
    
    try:
        # Chargement des données policières
        police_df = pd.read_sql_table(
            'statistiques_police', 
            engine
        )
        police_df = police_df[police_df['code_region'] == 32]
        
        # Chargement des données électorales avec vérification des colonnes
        election_2017_df = pd.read_sql(
            "SELECT * FROM election_2017 WHERE code_region = 32", 
            engine
        )
        
        election_2022_df = pd.read_sql(
            "SELECT * FROM election_2022 WHERE code_region = 32", 
            engine
        )
        
        # Standardisation des noms de colonnes
        column_mapping = {
            'Nom': 'nom',
            'Name': 'nom',
            'nom_candidat': 'nom',
            'Candidat': 'nom',
            '% Voix/Exp': 'pourcentage_voix_exprimes',
            'voix_exp': 'pourcentage_voix_exprimes',
            'pct_exprimes': 'pourcentage_voix_exprimes'
        }
        
        for df in [election_2017_df, election_2022_df]:
            df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
        
        return police_df, election_2017_df, election_2022_df
        
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        raise

def preprocess_data(police_df, election_2017_df, election_2022_df):
    """Prétraitement des données avec vérification des colonnes"""
    # Vérification des colonnes nécessaires
    for df, year in [(election_2017_df, 2017), (election_2022_df, 2022)]:
        if 'nom' not in df.columns:
            raise ValueError(f"Colonne 'nom' manquante dans les données {year}")
        if 'pourcentage_voix_exprimes' not in df.columns:
            raise ValueError(f"Colonne 'pourcentage_voix_exprimes' manquante dans les données {year}")
    
    # Conversion des taux criminels
    police_df['taux_pour_mille'] = pd.to_numeric(
        police_df['taux_pour_mille'].str.replace(',', '.'), 
        errors='coerce'
    )
    
    # Pivot des données criminelles
    crimes_pivot = police_df.pivot_table(
        index='annee', 
        columns='indicateur', 
        values='taux_pour_mille'
    ).dropna(axis=1, how='all')
    
    # Classification politique pour la droite
    right_parties = ['REPUBLICAIN', 'MACRON', 'LE PEN', 'RN', 'LR', 'LES REPUBLICAINS', 'RASSEMBLEMENT NATIONAL']
    
    def get_right_votes(df, year):
        try:
            # Vérification finale des colonnes
            if 'nom' not in df.columns or 'pourcentage_voix_exprimes' not in df.columns:
                raise ValueError("Colonnes nécessaires non trouvées")
                
            right_votes = df[
                df['nom'].str.contains('|'.join(right_parties), case=False, na=False)
            ]['pourcentage_voix_exprimes'].sum()
            
            return pd.DataFrame({'annee': [year], 'Droite': [right_votes]})
        except Exception as e:
            print(f"Erreur lors du traitement des données {year}: {str(e)}")
            raise
    
    # Traitement des élections
    election_2017_df['annee'] = 2017
    election_2022_df['annee'] = 2022
    
    elections_combined = pd.concat([
        get_right_votes(election_2017_df, 2017),
        get_right_votes(election_2022_df, 2022)
    ]).set_index('annee')
    
    return crimes_pivot, elections_combined

def plot_all_crime_indicators(crimes_pivot):
    """Graphique 1: Tous les indicateurs criminels"""
    plt.figure(figsize=(16, 10))
    
    # Palette de couleurs
    colors = plt.cm.tab20(np.linspace(0, 1, len(crimes_pivot.columns)))
    
    for i, crime in enumerate(crimes_pivot.columns):
        plt.plot(crimes_pivot.index, crimes_pivot[crime], 
                color=colors[i], marker='o', linestyle='--', linewidth=1.5,
                label=f'{crime[:20]}...' if len(crime) > 20 else crime)
    
    plt.title('Évolution complète des indicateurs criminels\nRégion 32', pad=20)
    plt.xlabel('Année', fontsize=12)
    plt.ylabel('Taux pour 1000 habitants', fontsize=12)
    plt.xticks(crimes_pivot.index)
    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_election_predictions(elections_df):
    """Graphique 2: Votes et prédictions présidentielles"""
    plt.figure(figsize=(12, 8))
    
    # Données historiques
    plt.plot(elections_df.index, elections_df['Droite'], 'ro-', 
            linewidth=3, markersize=12, label='Votes Droite (historique)')
    
    # Modèles de prédiction
    years = elections_df.index.values.reshape(-1, 1)
    votes = elections_df['Droite'].values
    
    # Linéaire
    lin_model = LinearRegression().fit(years, votes)
    lin_pred = max(0, min(100, lin_model.predict([[2027]])[0]))
    plt.plot(2027, lin_pred, 'gX', markersize=20, 
            label=f'2027 (Linéaire): {lin_pred:.1f}%')
    
    # Polynomial (deg 2)
    poly_model = make_pipeline(
        PolynomialFeatures(degree=2),
        LinearRegression()
    ).fit(years, votes)
    poly_pred = max(0, min(100, poly_model.predict([[2027]])[0]))
    plt.plot(2027, poly_pred, 'b*', markersize=20, 
            label=f'2027 (Polynomial): {poly_pred:.1f}%')
    
    # Tracé des tendances
    x_future = np.linspace(2017, 2027, 100)
    plt.plot(x_future, lin_model.predict(x_future.reshape(-1, 1)), 'g--', alpha=0.5)
    plt.plot(x_future, poly_model.predict(x_future.reshape(-1, 1)), 'b--', alpha=0.5)
    
    # Configuration finale
    plt.title('Prédiction des votes pour la droite\nÉlections présidentielles', pad=20)
    plt.xlabel('Année électorale', fontsize=12)
    plt.ylabel('Part des votes (%)', fontsize=12)
    plt.xticks([2017, 2022, 2027])
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    print("=== ANALYSE CRIMINALITÉ vs VOTES DROITE ===")
    
    try:
        # 1. Chargement
        print("\n1. Chargement des données...")
        police_df, election_2017_df, election_2022_df = load_data()
        
        # Afficher les colonnes disponibles pour débogage
        print("\nColonnes election_2017:", election_2017_df.columns.tolist())
        print("Colonnes election_2022:", election_2022_df.columns.tolist())
        
        # 2. Prétraitement
        print("\n2. Prétraitement des données...")
        crimes_pivot, elections_df = preprocess_data(police_df, election_2017_df, election_2022_df)
        
        # 3. Graphique des indicateurs criminels
        print("\n3. Génération du graphique des crimes...")
        plot_all_crime_indicators(crimes_pivot)
        
        # 4. Graphique des élections
        print("\n4. Génération du graphique électoral...")
        plot_election_predictions(elections_df)
        
    except Exception as e:
        print(f"\nERREUR: {str(e)}")
        raise
    
    print("\n=== ANALYSE TERMINÉE ===")

if __name__ == "__main__":
    main()