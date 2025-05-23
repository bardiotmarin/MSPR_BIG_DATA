import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine, text
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os
import sys
from pathlib import Path

# Configuration du chemin
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Importation des utilitaires
from src.utils import get_sqlalchemy_engine

# Configuration du style des graphiques
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
            
            # Chargement des données électorales
            election_2017_df = pd.read_sql(
                text("SELECT * FROM election_2017 WHERE code_region = 32"),
                conn
            )
            
            election_2022_df = pd.read_sql(
                text("SELECT * FROM election_2022 WHERE code_region = 32"),
                conn
            )
        
        # Standardisation des colonnes
        election_2017_df = standardize_columns(election_2017_df, year=2017)
        election_2022_df = standardize_columns(election_2022_df, year=2022)
        
        return police_df, election_2017_df, election_2022_df
        
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        raise

def standardize_columns(df, year):
    """Standardise les noms de colonnes des données électorales"""
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
    
    # Votes pour la droite (RN)
    right_parties = ['LE PEN', 'RN', 'RASSEMBLEMENT NATIONAL']
    
    def get_right_votes(df, year):
        right_votes = df[
            df['nom'].str.contains('|'.join(right_parties), case=False, na=False)
        ]['pourcentage_voix_exprimes'].sum()
        
        return pd.DataFrame({'annee': [year], 'votes_droite': [right_votes]})
    
    # Traitement des élections
    election_2017_df['annee'] = 2017
    election_2022_df['annee'] = 2022
    
    elections_combined = pd.concat([
        get_right_votes(election_2017_df, 2017),
        get_right_votes(election_2022_df, 2022)
    ]).set_index('annee')
    
    # Évolution des crimes avec gestion des NaN
    crimes_evolution = crimes_pivot.pct_change().mean(axis=1).fillna(0).to_frame('evolution_crimes')
    
    return crimes_pivot, elections_combined, crimes_evolution

def plot_all_crime_indicators(crimes_pivot, years_to_predict=4):
    """Graphique des indicateurs criminels avec prédictions, regroupés en figures multiples"""
    future_years = np.arange(crimes_pivot.index.max() + 1, crimes_pivot.index.max() + 1 + years_to_predict).reshape(-1, 1)
    
    # Liste des indicateurs criminels
    crime_indicators = list(crimes_pivot.columns)
    total_indicators = len(crime_indicators)
    
    # Nombre de sous-graphiques par figure (4 sous-graphiques par figure)
    plots_per_figure = 4
    num_figures = (total_indicators + plots_per_figure - 1) // plots_per_figure  # Arrondi au supérieur
    
    for fig_idx in range(num_figures):
        # Créer une nouvelle figure avec 4 sous-graphiques (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()  # Aplatir pour une indexation plus facile
        
        # Plage des indicateurs pour cette figure
        start_idx = fig_idx * plots_per_figure
        end_idx = min(start_idx + plots_per_figure, total_indicators)
        
        for idx, crime_idx in enumerate(range(start_idx, end_idx)):
            crime_type = crime_indicators[crime_idx]
            series = crimes_pivot[crime_type].dropna()
            
            if len(series) < 3:
                print(f"Pas assez de données pour {crime_type}")
                axes[idx].set_visible(False)
                continue

            X = series.index.values.reshape(-1, 1)
            y = series.values

            # Modèles de prédiction
            models = {
                'Linéaire': LinearRegression(),
                'Quadratique': make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
            }

            preds = {}
            for name, model in models.items():
                try:
                    model.fit(X, y)
                    preds[name] = model.predict(future_years)
                except Exception as e:
                    print(f"Erreur modèle {name} pour {crime_type}: {e}")
                    preds[name] = np.nan * np.ones(len(future_years))

            # Modèle ARIMA
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    arima = ARIMA(y, order=(1,1,1)).fit()
                    arima_pred = arima.forecast(steps=years_to_predict)
                    preds['Arima'] = arima_pred.values if hasattr(arima_pred, 'values') else arima_pred
            except Exception as e:
                print(f"Erreur modèle ARIMA pour {crime_type}: {e}")
                preds['Arima'] = np.nan * np.ones(years_to_predict)

            # Tracer les données historiques
            axes[idx].plot(X, y, 'ko-', label='Historique')

            # Tracer les prédictions
            colors = {'Linéaire': 'purple', 'Quadratique': 'red', 'Arima': 'green'}
            styles = {'Linéaire': '--', 'Quadratique': '-.', 'Arima': ':'}
            for name, values in preds.items():
                if not np.isnan(values).all():
                    axes[idx].plot(future_years, values, styles[name], color=colors[name], label=name)

            # Configuration du sous-graphique
            axes[idx].set_title(f"Prédictions pour {crime_type}", fontsize=10)
            axes[idx].set_xlabel('Année')
            axes[idx].set_ylabel('Taux pour mille')
            axes[idx].legend(fontsize=8)
            axes[idx].grid(True, alpha=0.3)

        # Ajuster l'espacement et afficher la figure
        plt.suptitle(f"Prédictions des indicateurs criminels - Groupe {fig_idx + 1}", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajuster pour le titre global
        plt.show()

def plot_combined_predictions(elections_df, crimes_evolution):
    """Prédictions combinées votes et crimes"""
    plt.close('all')
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # ===== PARTIE VOTES =====
    ax1.plot(elections_df.index, elections_df['votes_droite'], 'ro-', 
            linewidth=3, markersize=10, label='Votes RN (historique)')
    
    # Modèles de prédiction
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
    
    # ===== PARTIE CRIMES =====
    ax2 = ax1.twinx()
    
    # Conversion en pourcentage
    crimes_evolution *= 100
    
    ax2.plot(crimes_evolution.index, crimes_evolution['evolution_crimes'], 'bs-', 
            linewidth=2, markersize=8, label='Évolution crimes (historique)')
    
    # Prédiction crimes si assez de données
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
    
    # ===== CONFIGURATION FINALE =====
    plt.title('Prédictions combinées: Votes RN et Évolution des crimes\nRégion 32 (jusqu\'en 2027)', 
             pad=20, fontsize=14)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.xticks(np.append(np.unique(np.append(elections_df.index, crimes_evolution.index)), 2027))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    print("=== ANALYSE CRIMINALITÉ vs VOTES RN ===")
    
    try:
        # Chargement des données
        police_df, election_2017_df, election_2022_df = load_data()
        
        # Prétraitement
        crimes_pivot, elections_df, crimes_evolution = preprocess_data(
            police_df, election_2017_df, election_2022_df
        )
        
        print("\nDonnées électorales:")
        print(elections_df)
        print("\nÉvolution des crimes (%):")
        print(crimes_evolution * 100)
        
        # Graphiques
        plot_all_crime_indicators(crimes_pivot)
        plot_combined_predictions(elections_df, crimes_evolution)
        
    except Exception as e:
        print(f"\nERREUR: {str(e)}")
    
    print("\n=== ANALYSE TERMINÉE ===")

if __name__ == "__main__":
    main()