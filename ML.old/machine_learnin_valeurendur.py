import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import warnings

# Configuration du style
plt.style.use('ggplot')
sns.set_palette("husl")
warnings.filterwarnings("ignore")

# Valeurs réalistes pour le Gers (estimations historiques)
EXPECTED_EXPRIMES = {
    2002: 115000,  # Environ 115,000 exprimés en 2002
    2007: 112000,  # Environ 112,000 exprimés en 2007
    2012: 111000,  # Environ 111,000 exprimés en 2012
    2017: 110000,  # Environ 110,000 exprimés en 2017
    2022: 108000   # Environ 108,000 exprimés en 2022
}

# Pourcentages réels pour 2002 (approximatifs pour le Gers, 1er tour)
PERCENTAGES_2002 = {
    'CHIRAC': 22.0,
    'LE PEN': 15.0,
    'JOSPIN': 16.0,
    'BAYROU': 7.0,
    'LAGUILLER': 5.0,
    'CHEVENEMENT': 5.0,
    'MAMERE': 5.0,
    'BESANCENOT': 4.0,
    'HUE': 4.0
}

# Pourcentages réels pour 2007 (approximatifs pour le Gers, 1er tour)
PERCENTAGES_2007 = {
    'SARKOZY': 27.0,
    'ROYAL': 25.0,
    'BAYROU': 18.0,
    'LE PEN': 10.0,
    'BESANCENOT': 4.0,
    'BUFFET': 2.0,
    'VOYNET': 2.0,
    'DE VILLIERS': 2.0,
    'LAGUILLER': 1.0
}

# Pourcentages réels pour 2012 (approximatifs pour le Gers, 1er tour)
PERCENTAGES_2012 = {
    'HOLLANDE': 28.0,
    'SARKOZY': 25.0,
    'LE PEN': 15.0,
    'MELENCHON': 12.0,
    'BAYROU': 9.0,
    'JOLY': 3.0,
    'DUPONTAIGNAN': 2.0,
    'POUTOU': 1.0,
    'ARTHAUD': 1.0
}

# Pourcentages réels pour 2017 (approximatifs pour le Gers, 1er tour)
PERCENTAGES_2017 = {
    'MACRON': 25.0,
    'LE PEN': 18.0,
    'MELENCHON': 22.0,
    'FILLON': 18.0,
    'HAMON': 8.0,
    'DUPONTAIGNAN': 4.5,
    'LASSALLE': 4.0,
    'POUTOU': 1.0,
    'ARTHAUD': 0.6,
    'ASSELINEAU': 0.9,
    'CHEMINADE': 0.2
}

# Pourcentages réels pour 2022 (approximatifs pour le Gers, 1er tour)
PERCENTAGES_2022 = {
    'MACRON': 28.0,
    'LE PEN': 23.0,
    'MELENCHON': 22.0,
    'ZEMMOUR': 5.0,
    'PECRESSE': 4.0,
    'JADOT': 4.0,
    'HIDALGO': 2.0,
    'ROUSSEL': 2.0,
    'ARTHAUD': 1.0,
    'LASSALLE': 3.0,
    'POUTOU': 1.0,
    'DUPONTAIGNAN': 5.0
}

def load_data():
    """Charge les données électorales à partir des pourcentages bruts"""
    election_dfs = {}
    years = [2002, 2007, 2012, 2017, 2022]
    
    for year in years:
        # Récupérer les pourcentages et exprimés attendus
        percentages = globals()[f'PERCENTAGES_{year}']
        expected_exprimes = EXPECTED_EXPRIMES[year]
        
        # Créer un DataFrame avec les données brutes
        data = []
        for nom, percentage in percentages.items():
            voix = (percentage * expected_exprimes) / 100
            data.append({
                'Nom': nom,
                'voix': voix,
                '% Voix/Exp': percentage,
                'Exprimés': expected_exprimes if year != 2022 else None
            })
        
        df = pd.DataFrame(data)
        print(f"\nDonnées brutes pour {year} :")
        print(df)
        
        # Standardiser les colonnes
        df = standardize_columns(df, year)
        
        # Agrégation (simplifiée, juste pour uniformité)
        df = df.groupby('nom').agg({
            'voix': 'sum'
        }).reset_index()
        
        print(f"\nDonnées election_{year}_df après standardisation et agrégation :")
        print(df)
        election_dfs[year] = df
    
    return election_dfs

def transform_election_df(df, year):
    """Transforme un DataFrame électoral de format large à format long"""
    # Cette fonction n'est plus nécessaire avec les données brutes
    return df

def standardize_columns(df, year):
    """Standardise les noms de colonnes"""
    column_mapping = {
        2002: {
            'Sexe': 'sexe',
            'Nom': 'nom',
            'Prénom': 'prenom',
            'Voix': 'voix',
            '% Voix/Exp': 'pourcentage_voix_exprimes'
        },
        2007: {
            'Sexe': 'sexe',
            'Nom': 'nom',
            'Prénom': 'prenom',
            'Voix': 'voix',
            '% Voix/Exp': 'pourcentage_voix_exprimes'
        },
        2012: {
            'Sexe': 'sexe',
            'Nom': 'nom',
            'Prénom': 'prenom',
            'Voix': 'voix',
            '% Voix/Exp': 'pourcentage_voix_exprimes'
        },
        2017: {
            'Sexe': 'sexe',
            'Nom': 'nom',
            'Prénom': 'prenom',
            'Voix': 'voix',
            '% Voix/Exp': 'pourcentage_voix_exprimes'
        },
        2022: {
            'Nom': 'nom',
            'Voix': 'voix',
            '% Voix/Exp': 'pourcentage_voix_exprimes'
        }
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping[year].items() if k in df.columns})
    
    # Vérifier si la colonne 'nom' existe avant de normaliser
    if 'nom' in df.columns:
        # S'assurer que la colonne 'nom' est de type string
        df['nom'] = df['nom'].astype(str).fillna('')
        # Normaliser : supprimer espaces, majuscules, accents
        df['nom'] = df['nom'].str.strip().str.upper()
        # Remplacer les accents pour améliorer la détection
        df['nom'] = df['nom'].str.replace('É', 'E').str.replace('È', 'E').str.replace('Ê', 'E')
        df['nom'] = df['nom'].str.replace('À', 'A').str.replace('Â', 'A')
        df['nom'] = df['nom'].str.replace('Ç', 'C')
        df['nom'] = df['nom'].str.replace('Ô', 'O')
        df['nom'] = df['nom'].str.replace('Ù', 'U').str.replace('Û', 'U')
        # Gérer les tirets, espaces multiples, et variations
        df['nom'] = df['nom'].str.replace('-', ' ').str.replace('  ', ' ')
        df['nom'] = df['nom'].str.replace('DUPONT AIGNAN', 'DUPONTAIGNAN')
    else:
        print(f"Avertissement : Colonne 'nom' manquante dans le DataFrame pour l'année {year}")
    
    return df

def analyze_election_results(election_dfs):
    """Analyse détaillée des résultats par parti (en nombre de voix)"""
    parties = {
        'RN': ['LE PEN', 'MARINE', 'RN', 'RASSEMBLEMENT', 'NATIONAL', 'FN'],
        'LREM': ['MACRON', 'EMMANUEL', 'LREM', 'PRESIDENT', 'RENAISSANCE', 'ENSEMBLE'],
        'LR': ['LES REPUBLICAINS', 'REPUBLICAIN', 'LR', 'PECRESSE', 'CIOTTI', 'SARKOZY', 'FILLON', 'CHIRAC', 'RPR', 'UMP'],
        'LFI': ['MELENCHON', 'JEAN-LUC', 'LFI', 'FRANCE INSOMISE', 'FG'],
        'PS': ['PS', 'SOCIALISTE', 'HAMON', 'HOLLANDE', 'HIDALGO', 'JOSPIN', 'ROYAL'],
        'ECOLO': ['JADOT', 'ECOLOGIE', 'VERT', 'EELV', 'MAMERE', 'VOYNET', 'JOLY'],
        'REC': ['DUPONT-AIGNAN', 'DUPONTAIGNAN', 'RECONQUETE', 'ZEMMOUR', 'DUPONT AIGNAN', 'DE VILLIERS', 'MPF'],
        'PCF': ['ROUSSEL', 'COMMUNISTE', 'PCF', 'HUE', 'BUFFET'],
        'LO': ['ARTHAUD', 'LUTTE OUVRIERE', 'LAGUILLER'],
        'CENTRE': ['BAYROU', 'UDF', 'MODEM'],
        'AUTRES': []
    }
    
    results = {}
    for year, df in election_dfs.items():
        year_results = {}
        if 'voix' not in df.columns:
            print(f"Avertissement : Colonne 'voix' manquante pour l'année {year}")
            total_voices = 0
        else:
            total_voices = df['voix'].sum()  # Total des voix
        
        # Débogage : Afficher tous les noms dans le DataFrame
        if 'nom' in df.columns:
            print(f"\nNoms des candidats ({year}) :")
            print(df['nom'].unique())
        else:
            print(f"Avertissement : Colonne 'nom' manquante pour l'année {year}")
        
        # Vérifier la somme des voix exprimées
        print(f"Total des voix ({year}) : {total_voices:.0f}")
        
        assigned_voices = 0
        assigned_candidates = []
        for party, keywords in parties.items():
            if keywords:
                if 'nom' not in df.columns:
                    year_results[party] = 0
                    continue
                # S'assurer que la colonne nom est de type string
                df['nom'] = df['nom'].astype(str).fillna('')
                mask = df['nom'].str.contains('|'.join(keywords), case=False, na=False)
                voices = df[mask]['voix'].sum() if 'voix' in df.columns else 0
                year_results[party] = voices
                assigned_voices += voices
                # Ajouter les candidats détectés
                assigned_candidates.extend(df[mask]['nom'].unique())
            else:
                year_results[party] = 0
        
        # Calculer les voix non attribuées (AUTRES)
        year_results['AUTRES'] = max(0, total_voices - assigned_voices)
        
        # Débogage : Afficher les candidats non attribués (ceux dans AUTRES)
        if 'nom' in df.columns:
            unassigned_candidates = df[~df['nom'].isin(assigned_candidates)]['nom'].unique()
            print(f"\nCandidats non attribués (comptés dans AUTRES) pour {year} :")
            print(unassigned_candidates)
        
        results[year] = year_results
    
    return pd.DataFrame(results).T

def predict_party_voices(election_results, years_to_predict=[2027, 2032]):
    """Prédiction de l'évolution des voix des partis"""
    # Vérification des données
    print("\n🔍 Vérification des données électorales (voix) :")
    print(election_results)
    
    if election_results.isnull().values.any():
        election_results = election_results.fillna(0)
    
    predictions = {}
    historical_years = election_results.index.values  # [2002, 2007, 2012, 2017, 2022]
    all_years = np.concatenate([historical_years, years_to_predict])  # [2002, 2007, ..., 2027, 2032]
    years_for_fit = historical_years.reshape(-1, 1)  # Pour l'ajustement du modèle
    
    plt.figure(figsize=(14, 8))
    
    for i, party in enumerate(election_results.columns):
        if party == 'AUTRES':
            continue
            
        # Données historiques
        party_data = election_results[party].values
        model = make_pipeline(
            PolynomialFeatures(degree=2),
            LinearRegression()
        )
        
        try:
            # Ajuster le modèle polynomial sur les données historiques
            model.fit(years_for_fit, party_data)
            
            # Prédire pour 2027 et 2032
            future_years = np.array(years_to_predict).reshape(-1, 1)
            party_pred_future = model.predict(future_years)
            party_pred_future = np.clip(party_pred_future, 0, 120000)  # Limiter à 120,000 maximum
            
            # Stocker les prédictions
            for year, pred in zip(years_to_predict, party_pred_future):
                if year not in predictions:
                    predictions[year] = {}
                predictions[year][party] = max(0, pred)
            
            # Combiner les données historiques et prédites pour la période 2022-2032
            last_historical_year = historical_years[-1]  # 2022
            last_historical_value = party_data[-1]  # Valeur en 2022
            prediction_years = [last_historical_year] + years_to_predict  # [2022, 2027, 2032]
            prediction_values = [last_historical_value] + list(party_pred_future)  # [valeur 2022, pred 2027, pred 2032]
            
            # Tracer les lignes historiques (connecter les points historiques directement)
            plt.plot(historical_years, party_data, linestyle='-', alpha=0.7, color=f'C{i}', label=f'{party}')
            
            # Tracer la prédiction à partir de 2022
            plt.plot(prediction_years, prediction_values, linestyle='--', alpha=0.7, color=f'C{i}')
            
            # Points historiques
            plt.scatter(historical_years, party_data, color=f'C{i}', marker='o', s=100)
            
            # Points prédits
            plt.scatter(years_to_predict, party_pred_future, color=f'C{i}', marker='*', s=100)
        
        except Exception as e:
            print(f"Erreur pour {party}: {str(e)}")
            continue
    
    plt.title('Évolution des voix des partis politiques (2002-2032) - Gers (32, 1er tour)', pad=20)
    plt.xlabel('Année')
    plt.ylabel('Nombre de voix')
    plt.xticks(all_years)
    plt.ylim(0, 120000)  # Limiter à 120,000 pour le Gers
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('election_prediction_gers.png')  # Sauvegarde du graphique
    plt.show()
    
    print("\n🔮 Prédictions de l'évolution des voix par parti 🔮")
    for year in predictions:
        sorted_parties = sorted(predictions[year].items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 {year} - Parti dominant: {sorted_parties[0][0]} ({sorted_parties[0][1]:.0f} voix)")
        for party, voices in sorted_parties:
            print(f"  - {party}: {voices:.0f} voix")
    
    return predictions

def main():
    print("=== ANALYSE ET PRÉDICTIONS DES ÉLECTIONS PRÉSIDENTIELLES - GERS (32, 1er tour) ===")
    
    try:
        # 1. Chargement des données
        print("\n1. Chargement des données...")
        election_dfs = load_data()
        
        # 2. Analyse tous partis (en voix)
        print("\n2. Analyse des résultats électoraux (voix)...")
        election_results = analyze_election_results(election_dfs)
        print("\nRésultats électoraux (voix) :")
        print(election_results)
        
        # 3. Prédictions
        print("\n3. Préparation des prédictions (voix)...")
        predictions = predict_party_voices(election_results)
        
    except Exception as e:
        print(f"\nERREUR: {str(e)}")
    
    print("\n=== ANALYSE TERMINÉE ===")

if __name__ == "__main__":
    main()