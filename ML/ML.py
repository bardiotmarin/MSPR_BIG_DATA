import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings
import os
from dotenv import load_dotenv
from utils import get_sqlalchemy_engine 

# Chargement des variables d'environnement
load_dotenv()

# Configuration de la connexion PostgreSQL depuis les variables d'environnement
DB_CONFIG = {
    'host': 'localhost',  # Ou le nom du service 'postgres' si dans le même réseau Docker
    'database': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'port': '5433'  # Port mappé dans votre docker-compose
}

def connect_to_postgres():
    """Établit une connexion à PostgreSQL"""
    conn_str = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    engine = create_engine(conn_str)
    return engine

def load_data_from_postgres():
    """Charge les données de délinquance depuis PostgreSQL"""
    engine = get_sqlalchemy_engine()
    query = """
    SELECT * FROM statistiques_police 
    WHERE Code_region = 32 
    ORDER BY annee, indicateur
    """
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df

def preprocess_data(df):
    """Prétraitement des données"""
    # Conversion du taux pour mille en numérique (si nécessaire)
    if df['taux_pour_mille'].dtype == 'object':
        df['taux_pour_mille'] = df['taux_pour_mille'].str.replace(',', '.').astype(float)
    
    # Création d'un DataFrame pivot pour chaque indicateur
    pivot_df = df.pivot_table(index='annee', columns='indicateur', values='taux_pour_mille')
    return pivot_df

def train_models_and_predict(data, years_to_predict=3):
    """Entraîne des modèles et fait des prédictions"""
    predictions = {}
    last_year = data.index.max()
    future_years = np.array(range(last_year + 1, last_year + 1 + years_to_predict)).reshape(-1, 1)
    
    for crime_type in data.columns:
        series = data[crime_type].dropna()
        if len(series) < 3:  # Pas assez de données
            continue
            
        X = np.array(series.index).reshape(-1, 1)
        y = series.values
        
        # Modèle linéaire
        linear_model = LinearRegression()
        linear_model.fit(X, y)
        linear_pred = linear_model.predict(future_years)
        
        # Modèle polynomial (degré 2)
        poly_model = make_pipeline(PolynomialFeatures(2), LinearRegression())
        poly_model.fit(X, y)
        poly_pred = poly_model.predict(future_years)
        
        # Modèle ARIMA (simple)
        arima_pred = []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arima_model = ARIMA(series, order=(1,1,1))
                arima_model_fit = arima_model.fit()
                arima_pred = arima_model_fit.forecast(steps=years_to_predict)
        except:
            arima_pred = [np.nan] * years_to_predict
        
        # Moyenne des prédictions valides
        valid_preds = [pred for pred in [linear_pred, poly_pred, arima_pred] if not all(np.isnan(pred))]
        avg_pred = np.mean(valid_preds, axis=0) if valid_preds else [np.nan] * years_to_predict
        
        predictions[crime_type] = {
            'years': future_years.flatten(),
            'linear': linear_pred,
            'polynomial': poly_pred,
            'arima': arima_pred,
            'average': avg_pred,
            'history': series
        }
    
    return predictions

def plot_results(predictions):
    """Visualise les résultats avec matplotlib"""
    n_plots = len(predictions)
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    plt.figure(figsize=(18, 5 * n_rows))
    
    for i, (crime_type, data) in enumerate(predictions.items()):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Historique
        plt.plot(data['history'].index, data['history'].values, 'bo-', label='Historique')
        
        # Prédictions
        plt.plot(data['years'], data['linear'], 'r--', label='Linéaire')
        plt.plot(data['years'], data['polynomial'], 'g--', label='Polynomial')
        if not all(np.isnan(data['arima'])):
            plt.plot(data['years'], data['arima'], 'm--', label='ARIMA')
        plt.plot(data['years'], data['average'], 'k-', linewidth=2, label='Moyenne')
        
        plt.title(crime_type)
        plt.xlabel('Année')
        plt.ylabel('Taux pour mille')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def save_predictions_to_db(predictions):
    """Sauvegarde les prédictions dans PostgreSQL"""
    pred_df = pd.DataFrame()
    for crime_type, data in predictions.items():
        temp_df = pd.DataFrame({
            'indicateur': crime_type,
            'annee': data['years'],
            'prediction_taux': data['average'],
            'prediction_type': 'moyenne'
        })
        pred_df = pd.concat([pred_df, temp_df])
    
    engine = connect_to_postgres()
    pred_df.to_sql('predictions_delinquance', engine, if_exists='replace', index=False)
    engine.dispose()

def main():
    print("Début de l'analyse prédictive...")
    
    # 1. Chargement des données
    print("Chargement des données depuis PostgreSQL...")
    df = load_data_from_postgres()
    
    # 2. Prétraitement
    print("Prétraitement des données...")
    processed_data = preprocess_data(df)
    
    # 3. Modélisation et prédiction
    print("Entraînement des modèles et prédiction...")
    predictions = train_models_and_predict(processed_data)
    
    # 4. Visualisation
    print("Génération des visualisations...")
    plot_results(predictions)
    
    # 5. Sauvegarde
    print("Sauvegarde des résultats dans PostgreSQL...")
    save_predictions_to_db(predictions)
    
    print("Analyse terminée avec succès!")

if __name__ == "__main__":
    main()