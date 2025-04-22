import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
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
    """Charge les données depuis PostgreSQL avec gestion flexible des colonnes"""
    engine = get_sqlalchemy_engine()
    
    try:
        # Chargement des données policières
        with engine.connect() as conn:
            police_df = pd.read_sql(
                text("SELECT * FROM statistiques_police WHERE code_region = 32 ORDER BY annee, indicateur"),
                conn
            )
            
            # Chargement des données électorales avec détection automatique des colonnes
            election_2017_df = pd.read_sql(
                text("SELECT * FROM election_2017 WHERE code_region = 32"),
                conn
            )
            
            election_2022_df = pd.read_sql(
                text("SELECT * FROM election_2022 WHERE code_region = 32"),
                conn
            )
        
        # Standardisation des noms de colonnes pour 2017
        election_2017_df = standardize_columns(election_2017_df, year=2017)
        
        # Standardisation des noms de colonnes pour 2022
        election_2022_df = standardize_columns(election_2022_df, year=2022)
        
        return police_df, election_2017_df, election_2022_df
        
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        raise

def standardize_columns(df, year):
    """Standardise les noms de colonnes avec gestion des variantes"""
    column_mapping = {
        2017: {
            'Sexe': 'sexe',
            'Nom': 'nom',
            'Prénom': 'prenom',
            'Voix': 'voix',
            '% Voix/Exp': 'pourcentage_voix_exprimes',
            'Voix/Exp': 'pourcentage_voix_exprimes'
        },
        2022: {
            'Nom': 'nom',
            'nom_candidat': 'nom',
            'Candidat': 'nom',
            '% Voix/Exp': 'pourcentage_voix_exprimes',
            'pct_exprimes': 'pourcentage_voix_exprimes'
        }
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping[year].items() if k in df.columns})
    
    required_cols = ['nom', 'pourcentage_voix_exprimes']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\nATTENTION: Colonnes manquantes ({year}): {missing_cols}")
        print("Colonnes disponibles:", df.columns.tolist())
    
    return df

def preprocess_data(police_df, election_2017_df, election_2022_df):
    """Prétraitement des données pour l'analyse"""
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
    
    # Classification politique pour la droite (focus sur RN)
    right_parties = ['LE PEN', 'RN', 'RASSEMBLEMENT NATIONAL']
    
    def get_right_votes(df, year):
        nom_col = 'nom' if 'nom' in df.columns else next((col for col in df.columns if 'nom' in col.lower()), None)
        voix_col = 'pourcentage_voix_exprimes' if 'pourcentage_voix_exprimes' in df.columns else next((col for col in df.columns if 'voix' in col.lower() and 'exp' in col.lower()), None)
        
        if not nom_col or not voix_col:
            available_cols = df.columns.tolist()
            raise ValueError(f"Colonnes nécessaires non trouvées (année {year}).\nColonnes disponibles: {available_cols}")
            
        right_votes = df[
            df[nom_col].str.contains('|'.join(right_parties), case=False, na=False)
        ][voix_col].sum()
        
        return pd.DataFrame({'annee': [year], 'votes_droite': [right_votes]})
    
    # Traitement des élections
    election_2017_df['annee'] = 2017
    election_2022_df['annee'] = 2022
    
    elections_combined = pd.concat([
        get_right_votes(election_2017_df, 2017),
        get_right_votes(election_2022_df, 2022)
    ]).set_index('annee')
    
    # Calcul de l'évolution des crimes
    crimes_evolution = crimes_pivot.pct_change().mean(axis=1).to_frame('evolution_crimes')
    
    # Combinaison des données
    combined_data = elections_combined.join(crimes_evolution, how='outer').dropna()
    
    return crimes_pivot, elections_combined, combined_data

def plot_all_crime_indicators(crimes_pivot):
    """Graphique 1: Tous les indicateurs criminels"""
    plt.figure(figsize=(16, 10))
    
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

def plot_crime_vs_votes(combined_data):
    """Graphique 2: Corrélation entre crimes et votes"""
    plt.figure(figsize=(12, 8))
    
    # Création d'un graphique avec deux axes y
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Axe 1: Votes droite
    color = 'tab:red'
    ax1.set_xlabel('Année')
    ax1.set_ylabel('Votes RN (%)', color=color)
    ax1.plot(combined_data.index, combined_data['votes_droite'], 
             color=color, marker='o', label='Votes RN')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Axe 2: Évolution des crimes
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Évolution des crimes (%)', color=color)
    ax2.plot(combined_data.index, combined_data['evolution_crimes']*100, 
             color=color, marker='s', linestyle='--', label='Évolution crimes')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Titre et légende
    plt.title('Corrélation entre évolution des crimes et votes RN\nRégion 32', pad=20)
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_election_predictions(elections_df):
    """Graphique 3: Prédictions électorales pour 2027"""
    plt.figure(figsize=(12, 8))
    
    if len(elections_df) < 2:
        raise ValueError("Au moins 2 points de données nécessaires pour les prédictions")
    
    # Données historiques
    plt.plot(elections_df.index, elections_df['votes_droite'], 'ro-', 
            linewidth=3, markersize=12, label='Votes RN (historique)')
    
    # Préparation des données
    years = elections_df.index.values.reshape(-1, 1)
    votes = elections_df['votes_droite'].values
    
    # Modèles de prédiction
    models = {
        'Linéaire': LinearRegression(),
        'Polynomial (deg 2)': make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    }
    
    # Entraînement et prédiction
    predictions = {}
    for name, model in models.items():
        model.fit(years, votes)
        pred_2027 = model.predict([[2027]])[0]
        predictions[name] = {
            'value': max(0, min(100, pred_2027)),  # Contrainte entre 0% et 100%
            'trend': model.predict(np.linspace(min(years), 2027, 100).reshape(-1, 1))
        }
        
        # Affichage de la prédiction
        plt.plot(2027, predictions[name]['value'], 
                marker='X' if name == 'Linéaire' else '*', 
                markersize=20, 
                label=f'2027 ({name}): {predictions[name]["value"]:.1f}%')
        
        # Tracé de la tendance
        plt.plot(np.linspace(min(years), 2027, 100), 
                predictions[name]['trend'], 
                '--', alpha=0.5)
    
    # Configuration des axes
    plt.xticks(np.append(elections_df.index, 2027))
    plt.ylim(0, max(elections_df['votes_droite'].max() * 1.5, 50))
    
    # Titres et légendes
    plt.title('Prédiction des votes RN pour 2027\nRégion 32', pad=20)
    plt.xlabel('Année électorale', fontsize=12)
    plt.ylabel('Part des votes (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    print("=== ANALYSE CRIMINALITÉ vs VOTES RN ===")
    
    try:
        # 1. Chargement
        print("\n1. Chargement des données...")
        police_df, election_2017_df, election_2022_df = load_data()
        
        # 2. Prétraitement
        print("\n2. Prétraitement des données...")
        crimes_pivot, elections_df, combined_data = preprocess_data(police_df, election_2017_df, election_2022_df)
        
        print("\nRésumé des votes RN:")
        print(elections_df)
        print("\nDonnées combinées crimes/votes:")
        print(combined_data)
        
        # 3. Graphique des indicateurs criminels
        print("\n3. Génération du graphique des crimes...")
        plot_all_crime_indicators(crimes_pivot)
        
        # 4. Graphique de corrélation crimes/votes
        print("\n4. Génération du graphique de corrélation...")
        plot_crime_vs_votes(combined_data)
        
        # 5. Graphique des prédictions électorales
        print("\n5. Génération des prédictions électorales...")
        plot_election_predictions(elections_df)
        
    except Exception as e:
        print(f"\nERREUR: {str(e)}")
        raise
    
    print("\n=== ANALYSE TERMINÉE ===")

if __name__ == "__main__":
    main()