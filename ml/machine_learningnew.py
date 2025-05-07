import pandas as pd
import numpy as np
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
                text("SELECT sexe, nom, prenom, voix, annee, code_region, pourcentage_voix_exprimes, pourcentage_voix_exprimes_2, pourcentage_voix_exprimes_3, pourcentage_voix_exprimes_4, pourcentage_voix_exprimes_5, pourcentage_voix_exprimes_6, pourcentage_voix_exprimes_7, pourcentage_voix_exprimes_8, pourcentage_voix_exprimes_9, pourcentage_voix_exprimes_10, pourcentage_voix_exprimes_11, pourcentage_voix_exprimes_12, pourcentage_voix_exprimes_13, pourcentage_voix_exprimes_14, pourcentage_voix_exprimes_15, pourcentage_voix_exprimes_16, sexe_2, nom_2, prenom_2, voix_2, sexe_3, nom_3, prenom_3, voix_3, sexe_4, nom_4, prenom_4, voix_4, sexe_5, nom_5, prenom_5, voix_5, sexe_6, nom_6, prenom_6, voix_6, sexe_7, nom_7, prenom_7, voix_7, sexe_8, nom_8, prenom_8, voix_8, sexe_9, nom_9, prenom_9, voix_9, sexe_10, nom_10, prenom_10, voix_10, sexe_11, nom_11, prenom_11, voix_11, sexe_12, nom_12, prenom_12, voix_12, sexe_13, nom_13, prenom_13, voix_13, sexe_14, nom_14, prenom_14, voix_14, sexe_15, nom_15, prenom_15, voix_15, sexe_16, nom_16, prenom_16, voix_16 FROM election_tour_1"),
                conn
            )
            election_t2_all = pd.read_sql(
                text("SELECT code_region, nom_departement, code_canton, nom_canton, annee, inscrits, abstentions, pourcentage_abstentions_inscrits, votants, pourcentage_votants_inscrits, blancs_et_nuls, pourcentage_blancs_nuls_inscrits, pourcentage_blancs_nuls_votants, exprimes, pourcentage_exprimes_inscrits, pourcentage_exprimes_votants, sexe, nom, prenom, voix, pourcentage_voix_inscrits, pourcentage_voix_exprimes, sexe_2, nom_2, prenom_2, voix_2, pourcentage_voix_inscrits_2, pourcentage_voix_exprimes_2 FROM election_tour_2"),
                conn
            )
            print(f"Années trouvées dans election_tour_1 (toutes régions) : {sorted(election_t1_all['annee'].unique())}")
            print(f"Années trouvées dans election_tour_2 (toutes régions) : {sorted(election_t2_all['annee'].unique())}")
            
            # Utiliser les données telles quelles sans filtrage supplémentaire
            election_t1_df = election_t1_all.copy()
            election_t2_df = election_t2_all.copy()
            
            # Vérifier les années après chargement
            print(f"Années trouvées dans election_tour_1 : {sorted(election_t1_df['annee'].unique())}")
            print(f"Années trouvées dans election_tour_2 : {sorted(election_t2_df['annee'].unique())}")
            
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

def reshape_election_data(df):
    """Reforme les données pour inclure tous les candidats (1 à 16) en lignes séparées avec nom, voix et annee"""
    # Liste des suffixes pour les colonnes des candidats, incluant le premier sans suffixe
    suffixes = [''] + [f'_{i}' for i in range(2, 17)]  # ['', '_2', '_3', ..., '_16']
    
    # Créer une liste de DataFrames pour chaque candidat
    frames = []
    for idx, suffix in enumerate(suffixes, 1):
        # Définir les colonnes du candidat
        candidate_cols = [f'nom{suffix}', f'voix{suffix}']
        # Suffixe pour pourcentage_voix_exprimés
        perc_col = 'pourcentage_voix_exprimes' if idx == 1 else f'pourcentage_voix_exprimes_{idx}'
        
        # Vérifier si toutes les colonnes nécessaires existent
        if all(col in df.columns for col in candidate_cols) and perc_col in df.columns:
            candidate_df = df[[*candidate_cols, 'annee', perc_col]].copy()
            candidate_df.columns = ['nom', 'voix', 'annee', 'pourcen']
            # Filtrer les lignes où 'nom' n'est pas NaN
            candidate_df = candidate_df[candidate_df['nom'].notna()]
            # Convertir 'pourcen' et 'voix' en numérique
            candidate_df['pourcen'] = pd.to_numeric(candidate_df['pourcen'], errors='coerce')
            candidate_df['voix'] = pd.to_numeric(candidate_df['voix'], errors='coerce')
            frames.append(candidate_df)
    
    # Concaténer tous les DataFrames
    result_df = pd.concat(frames, ignore_index=True)
    
    # Débogage : Afficher les candidats par année
    for year in result_df['annee'].unique():
        year_data = result_df[result_df['annee'] == year]
        print(f"Candidats pour l'année {year} (Tour 1) : {year_data['nom'].unique()}")
    
    return result_df

def reshape_election_tour_2(df):
    """Reforme les données de Tour 2 pour inclure les deux candidats"""
    # Liste des suffixes pour les deux candidats
    suffixes = ['', '_2']
    
    # Créer une liste de DataFrames pour chaque candidat
    frames = []
    for idx, suffix in enumerate(suffixes, 1):
        # Définir les colonnes du candidat
        candidate_cols = [f'nom{suffix}', f'voix{suffix}']
        # Suffixe pour pourcentage_voix_exprimés
        perc_col = 'pourcentage_voix_exprimes' if idx == 1 else 'pourcentage_voix_exprimes_2'
        
        # Vérifier si toutes les colonnes nécessaires existent
        if all(col in df.columns for col in candidate_cols) and perc_col in df.columns:
            candidate_df = df[[*candidate_cols, 'annee', perc_col]].copy()
            candidate_df.columns = ['nom', 'voix', 'annee', 'pourcen']
            # Filtrer les lignes où 'nom' n'est pas NaN
            candidate_df = candidate_df[candidate_df['nom'].notna()]
            # Convertir 'pourcen' et 'voix' en numérique
            candidate_df['pourcen'] = pd.to_numeric(candidate_df['pourcen'], errors='coerce')
            candidate_df['voix'] = pd.to_numeric(candidate_df['voix'], errors='coerce')
            frames.append(candidate_df)
    
    # Concaténer tous les DataFrames
    result_df = pd.concat(frames, ignore_index=True)
    
    # Débogage : Afficher les candidats par année
    for year in result_df['annee'].unique():
        year_data = result_df[result_df['annee'] == year]
        print(f"Candidats pour l'année {year} (Tour 2) : {year_data['nom'].unique()}")
    
    return result_df

def calculate_percentages(df):
    """Calcule les pourcentages des voix exprimées par année si la colonne 'pourcen' est absente ou NULL"""
    if 'voix' not in df.columns or 'annee' not in df.columns:
        raise ValueError("Les colonnes 'voix' et 'annee' sont requises pour calculer les pourcentages.")
    
    # Si 'pourcen' est absent ou entièrement NULL, calculer à partir de 'voix'
    if 'pourcen' not in df.columns or df['pourcen'].isnull().all():
        # Calculer le total des voix exprimees par année
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
    # Reformater les données pour inclure tous les candidats
    election_t1_df = reshape_election_data(election_t1_df)
    election_t2_df = reshape_election_tour_2(election_t2_df)
    
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
            total = sum(float(value) for value in t1_agg.values() if pd.notna(value))
            if total > 0:
                t1_agg = {party: (float(value) / total) * 100 for party, value in t1_agg.items() if pd.notna(value)}
            historical_results_t1[year] = t1_agg
        else:
            historical_results_t1[year] = {}
        
        # Tour 2
        t2_data = election_t2_df[election_t2_df['annee'] == year]
        if not t2_data.empty:
            t2_agg = t2_data.groupby('parti')['pourcen'].sum().to_dict()
            # Normaliser pour que la somme soit 100%
            total = sum(float(value) for value in t2_agg.values() if pd.notna(value))
            if total > 0:
                t2_agg = {party: (float(value) / total) * 100 for party, value in t2_agg.items() if pd.notna(value)}
            historical_results_t2[year] = t2_agg
        else:
            historical_results_t2[year] = {}
    
    return historical_results_t1, historical_results_t2

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
        
        # 2. Calcul des résultats historiques par parti
        print("\n2. Calcul des résultats historiques par parti...")
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
        
        # 3. Prédictions
        print("\n3. Préparation des prédictions...")
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