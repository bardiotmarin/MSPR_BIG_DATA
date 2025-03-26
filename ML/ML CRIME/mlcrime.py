import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Configuration du style
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['axes.grid'] = True

def load_and_prepare_data(filepath):
    """Charge et agrège les données par année"""
    try:
        df = pd.read_excel(filepath)
        
        # Vérification des colonnes nécessaires
        if 'annee' not in df.columns or 'nombre' not in df.columns:
            raise ValueError("Colonnes 'annee' ou 'nombre' manquantes")
        
        # Nettoyage et agrégation
        df = df.dropna(subset=['annee', 'nombre'])
        df['annee'] = df['annee'].astype(int)
        
        # Agrégation par année (somme de toutes les unités de compte)
        df_annual = df.groupby('annee')['nombre'].sum().reset_index()
        
        return df_annual.sort_values('annee')
    
    except Exception as e:
        print(f"Erreur lors du traitement des données: {str(e)}")
        return None

def plot_historical_trend(df):
    """Affiche l'évolution historique"""
    if df is None:
        return
    
    fig, ax = plt.subplots()
    
    # Graphique à barres pour les données réelles
    ax.bar(df['annee'], df['nombre'], 
           color='#2b8cbe', alpha=0.7, 
           label='Total annuel')
    
    # Ligne de tendance
    ax.plot(df['annee'], df['nombre'], 
            'o-', color='#e34a33', 
            linewidth=2, markersize=8)
    
    ax.set_title("Évolution du nombre total d'infractions par année", 
                 pad=20, fontsize=14)
    ax.set_xlabel("Année", fontsize=12)
    ax.set_ylabel("Nombre total d'infractions", fontsize=12)
    ax.legend()
    plt.xticks(df['annee'])
    plt.show()

def train_models_and_predict(df, years_to_predict=[2025, 2026, 2027]):
    """Entraîne les modèles et projette les tendances"""
    if df is None or len(df) < 3:
        print("Données insuffisantes pour la modélisation")
        return None
    
    X = df[['annee']].values
    y = df['nombre'].values
    
    # Création des modèles
    models = {
        'Linéaire': PolynomialFeatures(degree=1),
        'Quadratique': PolynomialFeatures(degree=2)
    }
    
    results = {}
    
    # Entraînement et prédiction
    for name, poly in models.items():
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        X_future = poly.transform(np.array(years_to_predict).reshape(-1, 1))
        results[name] = model.predict(X_future)
    
    # Visualisation
    fig, ax = plt.subplots()
    
    # Données historiques
    ax.bar(df['annee'], df['nombre'], 
           color='#2b8cbe', alpha=0.5, 
           label='Données réelles')
    
    # Prédictions
    colors = ['#fdbb84', '#31a354']
    for (name, preds), color in zip(results.items(), colors):
        ax.plot(years_to_predict, preds, 's--',
                color=color, linewidth=2,
                markersize=10, label=f'Projection {name}')
        
        # Annotation des valeurs prédites
        for year, val in zip(years_to_predict, preds):
            ax.text(year, val, f'{int(val):,}', 
                   ha='center', va='bottom',
                   fontsize=10, color=color)
    
    ax.set_title("Projection du nombre total d'infractions", 
                 pad=20, fontsize=14)
    ax.set_xlabel("Année", fontsize=12)
    ax.set_ylabel("Nombre d'infractions", fontsize=12)
    ax.legend(loc='upper left')
    
    # Ajustement des axes
    min_year = min(df['annee'].min(), min(years_to_predict))
    max_year = max(df['annee'].max(), max(years_to_predict))
    ax.set_xlim(min_year - 0.5, max_year + 0.5)
    
    plt.xticks(list(df['annee']) + years_to_predict)
    plt.tight_layout()
    plt.show()
    
    return results

def main():
    # Chemin vers votre fichier (à adapter)
    DATA_PATH = "C:\\Users\\droui\\Documents\\githubprojetcs\\MSPR_BIG_DATA\\DATA\\raw\\fichierdelinquancepresidence.xlsx"
    
    # Chargement des données
    df = load_and_prepare_data(DATA_PATH)
    
    if df is not None:
        print("\nRésumé des données agrégées par année:")
        print(df.to_string(index=False))
        
        # Affichage historique
        plot_historical_trend(df)
        
        # Modélisation et projection
        predictions = train_models_and_predict(df)
        
        # Affichage des résultats
        if predictions:
            print("\nProjections du nombre total d'infractions:")
            for year, (lin, quad) in zip([2025, 2026, 2027], 
                                        zip(predictions['Linéaire'], predictions['Quadratique'])):
                print(f"{year}:")
                print(f"  - Modèle linéaire: {int(lin):,}")
                print(f"  - Modèle quadratique: {int(quad):,}")

if __name__ == "__main__":
    main()