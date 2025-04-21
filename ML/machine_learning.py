import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from statsmodels.tsa.arima.model import ARIMA
import warnings
import os
import sys
from pathlib import Path

# Configuration du path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from src.utils import get_sqlalchemy_engine
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    print("Assurez-vous que:")
    print("1. Le fichier src/utils.py existe")
    print("2. Il contient une fonction get_sqlalchemy_engine()")
    sys.exit(1)

# Configuration du style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings("ignore")

def load_data_from_postgres():
    """Charge les données depuis PostgreSQL"""
    engine = get_sqlalchemy_engine()
    
    try:
        police_df = pd.read_sql(
            "SELECT * FROM statistiques_police WHERE code_region = 32 ORDER BY annee, indicateur", 
            engine
        )
        election_2017_df = pd.read_sql(
            "SELECT * FROM election_2017 WHERE code_region = 32", 
            engine
        )
        election_2022_df = pd.read_sql(
            "SELECT * FROM election_2022 WHERE code_region = 32", 
            engine
        )
        return police_df, election_2017_df, election_2022_df
        
    except Exception as e:
        print(f"Erreur lors du chargement: {e}")
        print("Tables disponibles:", 
              pd.read_sql("SELECT table_name FROM information_schema.tables WHERE table_schema='public'", engine))
        raise

def preprocess_data(police_df, election_2017_df, election_2022_df):
    """Prétraitement des données"""
    # Nettoyage données police
    if police_df['taux_pour_mille'].dtype == 'object':
        police_df['taux_pour_mille'] = police_df['taux_pour_mille'].str.replace(',', '.').astype(float)
    
    crimes_pivot = police_df.pivot_table(index='annee', columns='indicateur', values='taux_pour_mille')
    
    # Classification politique
    left_parties = ['SOCIALISTE', 'MELENCHON', 'COMMUNISTE', 'FI']
    right_parties = ['REPUBLICAIN', 'MACRON', 'LE PEN', 'RN', 'LR']
    
    # Gestion flexible des noms de colonnes
    nom_col_2017 = 'nom' if 'nom' in election_2017_df.columns else 'Nom'
    voix_col_2017 = '% Voix/Exp' if '% Voix/Exp' in election_2017_df.columns else 'pourcentage_voix_exprimes'
    
    nom_col_2022 = 'nom' if 'nom' in election_2022_df.columns else 'Nom'
    voix_col_2022 = '% Voix/Exp' if '% Voix/Exp' in election_2022_df.columns else 'pourcentage_voix_exprimes'
    
    # Ajout de la colonne année
    election_2017_df['annee'] = 2017
    election_2022_df['annee'] = 2022
    
    # Classification 2017
    election_2017_df['bloc'] = np.where(
        election_2017_df[nom_col_2017].str.contains('|'.join(left_parties), case=False, na=False),
        'Gauche',
        np.where(
            election_2017_df[nom_col_2017].str.contains('|'.join(right_parties), case=False, na=False),
            'Droite',
            'Autre'
        )
    )
    
    # Classification 2022
    election_2022_df['bloc'] = np.where(
        election_2022_df[nom_col_2022].str.contains('|'.join(left_parties), case=False, na=False),
        'Gauche',
        np.where(
            election_2022_df[nom_col_2022].str.contains('|'.join(right_parties), case=False, na=False),
            'Droite',
            'Autre'
        )
    )
    
    # Uniformisation et combinaison
    election_2017_df = election_2017_df.rename(columns={voix_col_2017: 'pourcentage_voix'})
    election_2022_df = election_2022_df.rename(columns={voix_col_2022: 'pourcentage_voix'})
    
    elections_combined = pd.concat([
        election_2017_df[['annee', 'bloc', 'pourcentage_voix']],
        election_2022_df[['annee', 'bloc', 'pourcentage_voix']]
    ])
    
    elections_pivot = elections_combined.groupby(['annee', 'bloc'])['pourcentage_voix'].mean().unstack()
    
    return crimes_pivot, elections_pivot

def analyze_correlations(crimes_pivot, elections_pivot):
    """Analyse des corrélations"""
    crime_rate = crimes_pivot.mean(axis=1).rename('taux_crime')
    merged = pd.merge(crime_rate.to_frame(), elections_pivot, left_index=True, right_index=True, how='inner')
    
    if len(merged) < 2:
        print("\nPas assez de données pour calculer les corrélations")
        return None
    
    # Calcul des corrélations
    corr_left = merged['taux_crime'].corr(merged['Gauche']) if 'Gauche' in merged.columns else np.nan
    corr_right = merged['taux_crime'].corr(merged['Droite']) if 'Droite' in merged.columns else np.nan
    
    print(f"\nCorrélations:")
    print(f"Gauche: {corr_left:.3f}" if not np.isnan(corr_left) else "Gauche: Données insuffisantes")
    print(f"Droite: {corr_right:.3f}" if not np.isnan(corr_right) else "Droite: Données insuffisantes")
    
    # Visualisation
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    if 'Gauche' in merged.columns:
        sns.regplot(x='taux_crime', y='Gauche', data=merged, ax=axes[0])
        axes[0].set_title(f'Corrélation avec la gauche (r = {corr_left:.2f})')
        for i, row in merged.iterrows():
            axes[0].text(row['taux_crime'], row['Gauche'], str(i))
    
    if 'Droite' in merged.columns:
        sns.regplot(x='taux_crime', y='Droite', data=merged, ax=axes[1])
        axes[1].set_title(f'Corrélation avec la droite (r = {corr_right:.2f})')
        for i, row in merged.iterrows():
            axes[1].text(row['taux_crime'], row['Droite'], str(i))
    
    plt.tight_layout()
    plt.show()
    
    return merged

def predict_crime_trends(crimes_pivot, years_to_predict=4):
    """Prédictions des tendances"""
    results = {}
    future_years = np.arange(crimes_pivot.index.max()+1, crimes_pivot.index.max()+1+years_to_predict).reshape(-1, 1)
    
    for crime_type, series in crimes_pivot.items():
        series = series.dropna()
        if len(series) < 3:
            print(f"\nPas assez de données pour {crime_type}")
            continue
            
        X = series.index.values.reshape(-1, 1)
        y = series.values
        
        # Modèles
        models = {
            'linear': LinearRegression(),
            'quadratic': make_pipeline(PolynomialFeatures(2), LinearRegression())
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
                preds['arima'] = arima_pred.values if hasattr(arima_pred, 'values') else arima_pred
        except Exception as e:
            print(f"Erreur modèle ARIMA pour {crime_type}: {e}")
            preds['arima'] = np.nan * np.ones(years_to_predict)
        
        # Stockage résultats
        results[crime_type] = {
            '2027': {k: v[-1] for k, v in preds.items()},
            'all_years': {'years': future_years.flatten(), **preds}
        }
        
        # Visualisation
        plt.figure(figsize=(12, 6))
        plt.plot(X, y, 'ko-', label='Historique')
        
        colors = {'linear': 'blue', 'quadratic': 'green', 'arima': 'red'}
        for name, values in preds.items():
            if not np.isnan(values).all():
                plt.plot(future_years, values, '--', color=colors[name], label=name.capitalize())
        
        plt.title(f"Prédictions pour {crime_type}")
        plt.xlabel('Année')
        plt.ylabel('Taux pour mille')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return results

def main():
    print("=== Analyse Criminalité-Élections ===")
    
    try:
        # Chargement
        print("\nChargement des données...")
        police_df, election_2017_df, election_2022_df = load_data_from_postgres()
        print("Données chargées avec succès")
        
        # Prétraitement
        print("\nPrétraitement des données...")
        crimes_pivot, elections_pivot = preprocess_data(police_df, election_2017_df, election_2022_df)
        print("Prétraitement terminé")
        
        # Analyse
        print("\nAnalyse des corrélations...")
        merged_data = analyze_correlations(crimes_pivot, elections_pivot)
        
        # Prédictions
        print("\nGénération des prédictions...")
        predictions = predict_crime_trends(crimes_pivot)
        
        # Résultats
        if predictions:
            print("\nPrédictions pour 2027:")
            for crime, data in predictions.items():
                print(f"\n{crime}:")
                for model, value in data['2027'].items():
                    print(f"  {model.capitalize()}: {value:.2f}" if not np.isnan(value) else f"  {model.capitalize()}: Non disponible")
    
    except Exception as e:
        print(f"\nErreur lors de l'analyse: {str(e)}")
    
    print("\n=== Analyse terminée ===")

if __name__ == "__main__":
    main()