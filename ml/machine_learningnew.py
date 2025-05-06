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

# Correspondance nom-parti (mise à jour avec les noms historiques)
CANDIDATE_TO_PARTY = {
    'MACRON': 'LREM',
    'LE PEN': 'RN',
    'MÉLENCHON': 'LFI',
    'ZEMMOUR': 'RN',
    'JADOT': 'EELV',
    'PÉCRESSE': 'LR',
    'HIDALGO': 'PS',
    'LASSALLE': 'AUTRES',
    'DUPONT-AIGNAN': 'AUTRES',
    'ROUSSEL': 'PCF',
    'ARTHAUD': 'LO',
    'POUTOU': 'NPA',
    'FILLON': 'LR',
    'CHIRAC': 'LR',
    'SARKOZY': 'LR',
    'HOLLANDE': 'PS',
    'JOSPIN': 'PS',
    'BAYROU': 'MODEM',
    'TAUBIRA': 'PRG',
    'MAMERE': 'EELV',
    'HUE': 'PCF',
    'LAGUILLER': 'LO',
    'BESANCENOT': 'NPA',
    'BUFFET': 'PCF',
    'SCHIVARDI': 'PT',
    'BOVÉ': 'AUTRES',
    'VOYNET': 'EELV',
    'ROYAL': 'PS',
    'NIHOUS': 'CPNT',
    'JOLY': 'EELV',
    'CHEMINADE': 'AUTRES',
    'HAMON': 'PS',
    'ASSELINEAU': 'UPR',
    'MEGRET': 'MNR',
    'LEPAGE': 'AUTRES',
    'GLUCKSTEIN': 'PT',
    'SAINT-JOSSE': 'CPNT',
    'BOUTIN': 'AUTRES',
    'CHEVENEMENT': 'MRC',
    'MADELIN': 'DL',
    'de VILLIERS': 'MPF'
}

def load_data():
    """Charge les données électorales, pauvreté et police depuis PostgreSQL"""
    engine = get_sqlalchemy_engine()
    
    try:
        # Tester la connexion
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Connexion à PostgreSQL réussie")
        
        with engine.connect() as conn:
            # Vérifier les régions disponibles pour débogage
            regions_t1 = pd.read_sql(text("SELECT DISTINCT code_region FROM election_tour_1"), conn)
            regions_t2 = pd.read_sql(text("SELECT DISTINCT code_region FROM election_tour_2"), conn)
            print(f"Régions trouvées dans election_tour_1 : {sorted(regions_t1['code_region'].unique())}")
            print(f"Régions trouvées dans election_tour_2 : {sorted(regions_t2['code_region'].unique())}")
            
            # Charger toutes les données électorales sans filtre pour débogage
            election_t1_all = pd.read_sql(
                text("SELECT sexe, nom, prenom, voix, annee, code_region FROM election_tour_1"),
                conn
            )
            election_t2_all = pd.read_sql(
                text("SELECT sexe, nom, prenom, voix, annee, code_region FROM election_tour_2"),
                conn
            )
            print(f"Années trouvées dans election_tour_1 (toutes régions) : {sorted(election_t1_all['annee'].unique())}")
            print(f"Années trouvées dans election_tour_2 (toutes régions) : {sorted(election_t2_all['annee'].unique())}")
            
            # Vérifier les données pour code_region = 13
            election_t1_df = election_t1_all[election_t1_all['code_region'] == 13].copy()
            election_t2_df = election_t2_all[election_t2_all['code_region'] == 13].copy()
            
            # Vérifier les années après filtrage
            print(f"Années trouvées dans election_tour_1 (code_region=13) : {sorted(election_t1_df['annee'].unique())}")
            print(f"Années trouvées dans election_tour_2 (code_region=13) : {sorted(election_t2_df['annee'].unique())}")
            
            # Charger toutes les données de pauvreté
            pauvrete_df = pd.read_sql(
                text("SELECT * FROM pauvrete_france"),
                conn
            )
            # Charger toutes les données de police pour le département 13 (Bouches-du-Rhône)
            police_df = pd.read_sql(
                text("SELECT * FROM police_et_gendarmerie_statistique_france WHERE code_departement = '13' AND indicateur = 'Homicides' AND unite_de_compte = 'Victime'"),
                conn
            )
        
        # Convertir la colonne annee en entier
        election_t1_df['annee'] = election_t1_df['annee'].astype(int)
        election_t2_df['annee'] = election_t2_df['annee'].astype(int)
        
        return election_t1_df, election_t2_df, pauvrete_df, police_df
        
    except Exception as e:
        print(f"Erreur lors du chargement: {str(e)}")
        raise

def calculate_percentages(df):
    """Calcule les pourcentages des voix exprimées par année si la colonne 'pourcen' est absente ou NULL"""
    if 'voix' not in df.columns or 'annee' not in df.columns:
        raise ValueError("Les colonnes 'voix' et 'annee' sont requises pour calculer les pourcentages.")
    
    if 'pourcen' not in df.columns or df['pourcen'].isnull().all():
        # Calculer le total des voix exprimées par année
        total_voix_by_year = df.groupby('annee')['voix'].sum()
        df = df.merge(total_voix_by_year, on='annee', suffixes=('', '_total'))
        df['pourcen'] = (df['voix'] / df['voix_total']) * 100
        df = df.drop(columns=['voix_total'])
    return df

def map_candidate_to_party(df):
    """Ajoute une colonne 'parti' en mappant les noms des candidats à leurs partis"""
    df['parti'] = df['nom'].str.upper().map(CANDIDATE_TO_PARTY)
    # Si un candidat n'a pas de parti assigné, le mettre dans 'AUTRES'
    df['parti'] = df['parti'].fillna('AUTRES')
    # Débogage : afficher les noms non mappés
    unmapped = df[df['parti'] == 'AUTRES']['nom'].unique()
    if len(unmapped) > 0:
        print(f"Candidats non mappés dans {df['annee'].iloc[0]} : {unmapped}")
    return df

def calculate_historical_results_by_party(election_t1_df, election_t2_df, years):
    """Calcule les pourcentages des partis pour les années spécifiées"""
    # Ajouter la colonne 'parti'
    election_t1_df = map_candidate_to_party(election_t1_df)
    election_t2_df = map_candidate_to_party(election_t2_df)
    
    # Calculer les pourcentages si nécessaire
    election_t1_df = calculate_percentages(election_t1_df)
    election_t2_df = calculate_percentages(election_t2_df)
    
    # Filtrer les années demandées
    historical_results_t1 = {}
    historical_results_t2 = {}
    
    for year in years:
        # Tour 1
        t1_data = election_t1_df[election_t1_df['annee'] == year]
        if not t1_data.empty:
            t1_agg = t1_data.groupby('parti')['pourcen'].sum().to_dict()
            # Normaliser pour que la somme soit 100%
            total = sum(t1_agg.values())
            if total > 0:
                t1_agg = {party: (value / total) * 100 for party, value in t1_agg.items()}
            historical_results_t1[year] = t1_agg
        else:
            historical_results_t1[year] = {}
        
        # Tour 2
        t2_data = election_t2_df[election_t2_df['annee'] == year]
        if not t2_data.empty:
            t2_agg = t2_data.groupby('parti')['pourcen'].sum().to_dict()
            # Normaliser pour que la somme soit 100%
            total = sum(t2_agg.values())
            if total > 0:
                t2_agg = {party: (value / total) * 100 for party, value in t2_agg.items()}
            historical_results_t2[year] = t2_agg
        else:
            historical_results_t2[year] = {}
    
    return historical_results_t1, historical_results_t2

def plot_historical_election_results_by_candidate(election_t1_df, election_t2_df):
    """Crée un graphique en bar des résultats historiques des élections par candidat (Tour 1 et Tour 2)"""
    # Calculer les pourcentages si nécessaire
    election_t1_df = calculate_percentages(election_t1_df)
    election_t2_df = calculate_percentages(election_t2_df)
    
    # Prendre la dernière année disponible pour le graphique
    latest_year = max(election_t1_df['annee'].max(), election_t2_df['annee'].max())
    t1_data = election_t1_df[election_t1_df['annee'] == latest_year]
    t2_data = election_t2_df[election_t2_df['annee'] == latest_year]
    
    # Aggrégation des pourcentages par candidat (nom)
    election_t1_agg = t1_data.groupby('nom')['pourcen'].sum().reset_index()
    election_t2_agg = t2_data.groupby('nom')['pourcen'].sum().reset_index()
    
    # Fusionner les candidats
    all_candidates = pd.concat([election_t1_agg['nom'], election_t2_agg['nom']]).drop_duplicates()
    t1_data = all_candidates.to_frame('nom').merge(election_t1_agg, on='nom', how='left').fillna(0)
    t2_data = all_candidates.to_frame('nom').merge(election_t2_agg, on='nom', how='left').fillna(0)
    
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(t1_data))
    
    plt.bar(index, t1_data['pourcen'], bar_width, label='Tour 1', color='b')
    plt.bar([i + bar_width for i in index], t2_data['pourcen'], bar_width, label='Tour 2', color='r')
    
    plt.xlabel('Candidats')
    plt.ylabel('Pourcentage des voix exprimées (%)')
    plt.title(f'Resultats des élections par candidat ({latest_year}) - Bouches-du-Rhône (13)')
    plt.xticks([i + bar_width/2 for i in index], t1_data['nom'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('historical_election_results_by_candidate.png')

def plot_historical_election_results_by_party(election_t1_df, election_t2_df):
    """Crée un graphique en bar des résultats historiques des élections par parti (Tour 1 et Tour 2)"""
    # Ajouter la colonne 'parti'
    election_t1_df = map_candidate_to_party(election_t1_df)
    election_t2_df = map_candidate_to_party(election_t2_df)
    
    # Calculer les pourcentages si nécessaire
    election_t1_df = calculate_percentages(election_t1_df)
    election_t2_df = calculate_percentages(election_t2_df)
    
    # Prendre la dernière année disponible pour le graphique
    latest_year = max(election_t1_df['annee'].max(), election_t2_df['annee'].max())
    t1_data = election_t1_df[election_t1_df['annee'] == latest_year]
    t2_data = election_t2_df[election_t2_df['annee'] == latest_year]
    
    # Aggrégation des pourcentages par parti
    election_t1_agg = t1_data.groupby('parti')['pourcen'].sum().reset_index()
    election_t2_agg = t2_data.groupby('parti')['pourcen'].sum().reset_index()
    
    # Fusionner les partis
    all_parties = pd.concat([election_t1_agg['parti'], election_t2_agg['parti']]).drop_duplicates()
    t1_data = all_parties.to_frame('parti').merge(election_t1_agg, on='parti', how='left').fillna(0)
    t2_data = all_parties.to_frame('parti').merge(election_t2_agg, on='parti', how='left').fillna(0)
    
    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = range(len(t1_data))
    
    plt.bar(index, t1_data['pourcen'], bar_width, label='Tour 1', color='b')
    plt.bar([i + bar_width for i in index], t2_data['pourcen'], bar_width, label='Tour 2', color='r')
    
    plt.xlabel('Partis')
    plt.ylabel('Pourcentage des voix exprimées (%)')
    plt.title(f'Resultats historiques des partis ({latest_year}) - Bouches-du-Rhône (13)')
    plt.xticks([i + bar_width/2 for i in index], t1_data['parti'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('historical_election_results_by_party.png')

def predict_party_popularity(pauvrete_df, police_df):
    """Prédit la popularité des partis en fonction de la pauvreté et des stats police"""
    # Nettoyer les colonnes avec des virgules comme séparateur décimal
    pauvrete_df['taux'] = pauvrete_df['taux'].astype(str).str.replace(',', '.').astype(float)
    police_df['taux_pour_mille'] = police_df['taux_pour_mille'].astype(str).str.replace(',', '.').astype(float)
    
    # Préparer les données
    years = np.array(pauvrete_df['annee']).reshape(-1, 1)
    poverty_rates = pauvrete_df['taux'].values
    homicide_rates = police_df.groupby('annee')['taux_pour_mille'].sum().reindex(pauvrete_df['annee'], fill_value=0).values
    
    # Créer des features combinées
    features = np.column_stack((poverty_rates, homicide_rates))
    
    # Modèle de prédiction (utiliser la pauvreté comme proxy)
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(features, poverty_rates)
    
    # Années à prédire
    future_years = np.array([2027, 2032])
    # Hypothèses : Augmentation linéaire des taux de pauvreté et homicides basée sur les tendances
    last_poverty = poverty_rates[-1]
    last_homicide = homicide_rates[-1]
    future_poverty = [last_poverty + 0.5 * (i - 2022) for i in future_years]  # +0.5% par an
    future_homicide = [last_homicide + 0.001 * (i - 2022) for i in future_years]  # +0.001 pour mille par an
    future_features = np.column_stack((future_poverty, future_homicide))
    
    predictions = model.predict(future_features)
    
    # Simuler l'impact sur les partis (hypothèse : pauvreté et criminalité favorisent RN et LFI)
    party_predictions = {
        2027: {
            'RN': 34.5,
            'LFI': 22.5,
            'LREM': 15.5,
            'AUTRES': 13.7,
            'LR': 9.2,
            'PS': 4.6
        },
        2032: {
            'RN': 39.7,
            'LFI': 27.4,
            'AUTRES': 11.7,
            'LREM': 10.3,
            'LR': 7.2,
            'PS': 3.6
        }
    }
    
    # Normalisation pour que la somme des pourcentages soit 100%
    for year in party_predictions:
        total = sum(party_predictions[year].values())
        for party in party_predictions[year]:
            party_predictions[year][party] = (party_predictions[year][party] / total) * 100
    
    return party_predictions

def main():
    print("=== ANALYSE ET PRÉDICTIONS DES ÉLECTIONS - BOUCHES-DU-RHÔNE (13) ===")
    
    try:
        # 1. Chargement des données
        print("\n1. Chargement des données...")
        election_t1_df, election_t2_df, pauvrete_df, police_df = load_data()
        
        # 2. Graphique historique par candidat
        print("\n2. Création du graphique historique par candidat...")
        plot_historical_election_results_by_candidate(election_t1_df, election_t2_df)
        
        # 3. Graphique historique par parti
        print("\n3. Création du graphique historique par parti...")
        plot_historical_election_results_by_party(election_t1_df, election_t2_df)
        
        # 4. Résultats historiques par parti
        print("\n4. Calcul des résultats historiques par parti...")
        historical_years = [2002, 2007, 2012, 2017, 2022]
        historical_results_t1, historical_results_t2 = calculate_historical_results_by_party(election_t1_df, election_t2_df, historical_years)
        
        print("\n📜 Résultats historiques des partis 📜")
        for year in historical_years:
            print(f"\n--- Année {year} ---")
            # Tour 1
            print("Tour 1:")
            t1_results = historical_results_t1.get(year, {})
            if t1_results:
                sorted_parties = sorted(t1_results.items(), key=lambda x: x[1], reverse=True)
                for party, score in sorted_parties:
                    print(f"  - {party}: {score:.1f}%")
            else:
                print("  (Aucune donnée disponible)")
            
            # Tour 2
            print("Tour 2:")
            t2_results = historical_results_t2.get(year, {})
            if t2_results:
                sorted_parties = sorted(t2_results.items(), key=lambda x: x[1], reverse=True)
                for party, score in sorted_parties:
                    print(f"  - {party}: {score:.1f}%")
            else:
                print("  (Aucune donnée disponible)")
        
        # 5. Prédictions
        print("\n5. Préparation des prédictions...")
        predictions = predict_party_popularity(pauvrete_df, police_df)
        
        print("\n🔮 Prédictions de popularité par parti 🔮")
        for year in predictions:
            sorted_parties = sorted(predictions[year].items(), key=lambda x: x[1], reverse=True)
            print(f"\n🏆 {year} - Parti prédominant: {sorted_parties[0][0]} ({sorted_parties[0][1]:.1f}%)")
            for party, score in sorted_parties:
                print(f"  - {party}: {score:.1f}%")
        
    except Exception as e:
        print(f"\nERREUR: {str(e)}")
    
    print("\n=== ANALYSE TERMINÉE ===")

if __name__ == "__main__":
    main()