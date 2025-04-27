import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
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
    """Charge les données depuis PostgreSQL"""
    engine = get_sqlalchemy_engine()
    
    try:
        with engine.connect() as conn:
            police_df = pd.read_sql(
                text("SELECT * FROM statistiques_police WHERE code_region = 32 ORDER BY annee, indicateur"),
                conn
            )
            election_2017_df = pd.read_sql(
                text("SELECT * FROM election_2017 WHERE code_region = 32"),
                conn
            )
            election_2022_df = pd.read_sql(
                text("SELECT * FROM election_2022 WHERE code_region = 32"),
                conn
            )
        
        election_2017_df = standardize_columns(election_2017_df, year=2017)
        election_2022_df = standardize_columns(election_2022_df, year=2022)
        
        return police_df, election_2017_df, election_2022_df
        
    except Exception as e:
        print(f"Erreur lors du chargement: {str(e)}")
        raise

def standardize_columns(df, year):
    """Standardise les noms de colonnes"""
    column_mapping = {
        2017: {
            'Sexe': 'sexe',
            'Nom': 'nom',
            'Prénom': 'prenom',
            'Voix': 'voix',
            '% Voix/Exp': 'pourcentage_voix_exprimes'
        },
        2022: {
            'Nom': 'nom',
            '% Voix/Exp': 'pourcentage_voix_exprimes'
        }
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping[year].items() if k in df.columns})
    
    required_cols = ['nom', 'pourcentage_voix_exprimes']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Colonnes manquantes ({year}): {missing_cols}")
    
    return df

def preprocess_data(police_df, election_2017_df, election_2022_df):
    """Prétraitement des données"""
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
    
    # Évolution des crimes
    crimes_evolution = crimes_pivot.pct_change().mean(axis=1).fillna(0).to_frame('evolution_crimes')
    
    return crimes_pivot, crimes_evolution

def plot_all_crime_indicators(crimes_pivot):
    """Graphique des indicateurs criminels"""
    plt.figure(figsize=(16, 10))
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(crimes_pivot.columns)))
    
    for i, crime in enumerate(crimes_pivot.columns):
        plt.plot(crimes_pivot.index, crimes_pivot[crime], 
                color=colors[i], marker='o', linestyle='--', linewidth=1.5,
                label=f'{crime[:20]}...' if len(crime) > 20 else crime)
    
    plt.title('Évolution des indicateurs criminels\nRégion 32', pad=20)
    plt.xlabel('Année')
    plt.ylabel('Taux pour 1000 habitants')
    plt.xticks(crimes_pivot.index)
    plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def get_right_votes(election_df, year):
    """Récupère les votes RN"""
    right_parties = ['LE PEN', 'MARINE', 'RN', 'RASSEMBLEMENT', 'NATIONAL']
    votes = election_df[
        election_df['nom'].str.contains('|'.join(right_parties), case=False, na=False)
    ]['pourcentage_voix_exprimes'].sum()
    return pd.DataFrame({'annee': [year], 'votes_droite': [votes]})

def analyze_election_results(df_2017, df_2022):
    """Analyse détaillée des résultats par parti"""
    parties = {
        'RN': ['LE PEN', 'MARINE', 'RN', 'RASSEMBLEMENT', 'NATIONAL'],
        'LREM': ['MACRON', 'EMMANUEL', 'LREM', 'PRESIDENT', 'RENAISSANCE', 'ENSEMBLE'],
        'LR': ['LES REPUBLICAINS', 'REPUBLICAIN', 'LR', 'PECRESSE', 'CIOTTI', 'SARKOZY'],
        'LFI': ['MELENCHON', 'JEAN-LUC', 'LFI', 'FRANCE INSOMISE'],
        'PS': ['PS', 'SOCIALISTE', 'HAMON', 'OLAND'],
        'ECOLO': ['JADOT', 'ECOLOGIE', 'VERT', 'EELV'],
        'REC': ['DUPONT-AIGNAN', 'RECONQUETE', 'ZEMOUR'],
        'AUTRES': []
    }
    
    results = {}
    for year, df in [(2017, df_2017), (2022, df_2022)]:
        year_results = {}
        total = 0
        
        for party, keywords in parties.items():
            if keywords:
                mask = df['nom'].str.contains('|'.join(keywords), case=False, na=False)
                votes = df[mask]['pourcentage_voix_exprimes'].sum()
                year_results[party] = votes
                total += votes
            else:
                year_results[party] = 0
        
        if total > 0:
            for party in year_results:
                year_results[party] = year_results[party] * 100 / total
        
        results[year] = year_results
    
    return pd.DataFrame(results).T

def plot_combined_predictions(elections_df, crimes_evolution):
    """Prédictions combinées votes RN et crimes"""
    plt.close('all')
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Partie Votes RN
    ax1.plot(elections_df.index, elections_df['votes_droite'], 'ro-', 
            linewidth=3, markersize=10, label='Votes RN (historique)')
    
    years = elections_df.index.values.reshape(-1, 1)
    votes = elections_df['votes_droite'].values
    
    for name, model in [('Linéaire', LinearRegression()),
                       ('Polynomial', make_pipeline(PolynomialFeatures(degree=2), LinearRegression()))]:
        model.fit(years, votes)
        pred_2027 = max(10, min(60, model.predict([[2027]])[0]))
        x_future = np.linspace(min(years), 2027, 100)
        
        ax1.plot(2027, pred_2027, 
                marker='X' if name == 'Linéaire' else '*', 
                markersize=15, 
                label=f'Votes 2027 ({name}): {pred_2027:.1f}%')
        ax1.plot(x_future, model.predict(x_future), '--', alpha=0.4)

    ax1.set_xlabel('Année')
    ax1.set_ylabel('Votes RN (%)', color='red')
    ax1.tick_params(axis='y', labelcolor='red')
    ax1.set_ylim(0, 60)
    
    # Partie Crimes
    ax2 = ax1.twinx()
    crimes_evolution *= 100
    
    ax2.plot(crimes_evolution.index, crimes_evolution['evolution_crimes'], 'bs-', 
            linewidth=2, markersize=8, label='Évolution crimes (historique)')
    
    if len(crimes_evolution) >= 2:
        crime_model = LinearRegression().fit(
            crimes_evolution.index.values.reshape(-1, 1),
            crimes_evolution['evolution_crimes']
        )
        crime_pred_2027 = crime_model.predict([[2027]])[0]
        x_future_crimes = np.linspace(min(crimes_evolution.index), 2027, 100)
        
        ax2.plot(2027, crime_pred_2027, 'gD', markersize=12, 
                label=f'Crimes 2027: {crime_pred_2027:.1f}%')
        ax2.plot(x_future_crimes, crime_model.predict(x_future_crimes.reshape(-1, 1)), 'g--', alpha=0.4)
    
    ax2.set_ylabel('Évolution des crimes (%)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(-30, 30)
    
    plt.title('Prédictions combinées: Votes RN et Évolution des crimes\nRégion 32 (jusqu\'en 2027)', 
             pad=20, fontsize=14)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.xticks(np.append(np.unique(np.append(elections_df.index, crimes_evolution.index)), 2027))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def predict_party_popularity(election_results, years_to_predict=[2026, 2027, 2028, 2029]):
    """Prédiction de popularité des partis"""
    # Vérification des données
    print("\n🔍 Vérification des données électorales :")
    print(election_results)
    
    if election_results.isnull().values.any():
        election_results = election_results.fillna(0)
    
    predictions = {}
    years = election_results.index.values.reshape(-1, 1)
    
    plt.figure(figsize=(14, 8))
    
    for i, party in enumerate(election_results.columns):
        if party == 'AUTRES':
            continue
            
        model = make_pipeline(
            PolynomialFeatures(degree=2),
            LinearRegression()
        )
        
        try:
            model.fit(years, election_results[party])
            future_years = np.array(years_to_predict).reshape(-1, 1)
            party_pred = model.predict(future_years)
            
            for year, pred in zip(years_to_predict, party_pred):
                if year not in predictions:
                    predictions[year] = {}
                predictions[year][party] = max(1, min(50, pred))
            
            x_vals = np.linspace(min(years), max(years_to_predict), 100)
            plt.plot(x_vals, model.predict(x_vals), '--', alpha=0.3, color=f'C{i}')
            plt.scatter(years, election_results[party], label=f'{party} (historique)', color=f'C{i}')
            plt.scatter(years_to_predict, party_pred, marker='*', s=100, color=f'C{i}', label=f'{party} (prédiction)')
        
        except Exception as e:
            print(f"Erreur pour {party}: {str(e)}")
            continue
    
    plt.title('Prédiction de popularité des partis politiques (2026-2029)', pad=20)
    plt.xlabel('Année')
    plt.ylabel('Part des votes (%)')
    plt.xticks(np.append(election_results.index, years_to_predict))
    plt.ylim(0, 60)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n🔮 Prédictions de popularité par parti 🔮")
    for year in years_to_predict:
        total = sum(predictions[year].values())
        if total < 100:
            predictions[year]['AUTRES'] = 100 - total
        
        sorted_parties = sorted(predictions[year].items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 {year} - Parti prédominant: {sorted_parties[0][0]} ({sorted_parties[0][1]:.1f}%)")
        for party, score in sorted_parties:
            print(f"  - {party}: {score:.1f}%")
    
    return predictions

def main():
    print("=== ANALYSE CRIMINALITÉ vs VOTES POLITIQUES ===")
    
    try:
        # 1. Chargement des données
        print("\n1. Chargement des données...")
        police_df, election_2017_df, election_2022_df = load_data()
        
        # 2. Prétraitement
        print("\n2. Prétraitement des données...")
        crimes_pivot, crimes_evolution = preprocess_data(police_df, election_2017_df, election_2022_df)
        
        # 3. Analyse électorale RN
        elections_combined = pd.concat([
            get_right_votes(election_2017_df, 2017),
            get_right_votes(election_2022_df, 2022)
        ]).set_index('annee')
        
        # 4. Analyse tous partis
        election_results = analyze_election_results(election_2017_df, election_2022_df)
        print("\nRésultats électoraux normalisés :")
        print(election_results)
        
        # 5. Graphiques
        print("\n5. Génération des graphiques...")
        plot_all_crime_indicators(crimes_pivot)
        plot_combined_predictions(elections_combined, crimes_evolution)
        
        # 6. Prédictions
        print("\n6. Préparation des prédictions...")
        predictions = predict_party_popularity(election_results)
        
    except Exception as e:
        print(f"\nERREUR: {str(e)}")
    
    print("\n=== ANALYSE TERMINÉE ===")

if __name__ == "__main__":
    main()