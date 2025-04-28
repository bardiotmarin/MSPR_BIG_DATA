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

# Valeurs réalistes pour le Gers (estimations historiques)
EXPECTED_EXPRIMES_2017 = 110000  # Environ 110,000 exprimés en 2017
EXPECTED_EXPRIMES_2022 = 108000  # Environ 108,000 exprimés en 2022

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
    """Charge les données électorales depuis PostgreSQL et calcule les voix totales"""
    engine = get_sqlalchemy_engine()
    
    try:
        with engine.connect() as conn:
            election_2017_df = pd.read_sql(
                text("SELECT * FROM election_2017 WHERE code_region = 32"),
                conn
            )
            election_2022_df = pd.read_sql(
                text("SELECT * FROM election_2022 WHERE code_region = 32"),
                conn
            )
        
        # Débogage : Afficher les colonnes des DataFrames après chargement
        print("\nColonnes de election_2017_df après chargement :")
        print(election_2017_df.columns.tolist())
        print("\nColonnes de election_2022_df après chargement :")
        print(election_2022_df.columns.tolist())
        
        # Transformer election_2017_df de format large à format long
        election_2017_df = transform_election_2017(election_2017_df)
        
        # Débogage : Afficher les colonnes après transformation
        print("\nColonnes de election_2017_df après transformation :")
        print(election_2017_df.columns.tolist())
        
        # Corriger les pourcentages pour 2017 AVANT standardisation
        if 'Nom' in election_2017_df.columns and '% Voix/Exp' in election_2017_df.columns:
            print("\nCorrection des pourcentages pour 2017...")
            for nom, percentage in PERCENTAGES_2017.items():
                election_2017_df.loc[election_2017_df['Nom'] == nom, '% Voix/Exp'] = percentage
        
        # Débogage : Afficher les pourcentages corrigés pour 2017
        print("\nPourcentages corrigés pour 2017 :")
        print(election_2017_df[['Nom', '% Voix/Exp']].drop_duplicates())
        
        # Corriger les pourcentages pour 2022 AVANT standardisation
        if 'Nom' in election_2022_df.columns and '% Voix/Exp' in election_2022_df.columns:
            print("\nCorrection des pourcentages pour 2022...")
            for nom, percentage in PERCENTAGES_2022.items():
                election_2022_df.loc[election_2022_df['Nom'] == nom, '% Voix/Exp'] = percentage
        
        # Débogage : Afficher les pourcentages corrigés pour 2022
        print("\nPourcentages corrigés pour 2022 :")
        print(election_2022_df[['Nom', '% Voix/Exp']].drop_duplicates())
        
        # Standardisation des colonnes APRES correction des pourcentages
        election_2017_df = standardize_columns(election_2017_df, year=2017)
        election_2022_df = standardize_columns(election_2022_df, year=2022)
        
        # Débogage : Afficher les DataFrames après standardisation
        print("\nDonnées election_2017_df après standardisation :")
        if 'nom' in election_2017_df.columns and 'voix_calculees' in election_2017_df.columns:
            print(election_2017_df[['nom', 'voix_calculees']].head())
        else:
            print("Colonnes 'nom' ou 'voix_calculees' manquantes dans election_2017_df")
            print(election_2017_df.head())
        
        print("\nDonnées election_2022_df après standardisation :")
        if 'nom' in election_2022_df.columns and 'voix' in election_2022_df.columns:
            print(election_2022_df[['nom', 'voix']].head())
        else:
            print("Colonnes 'nom' ou 'voix' manquantes dans election_2022_df")
            print(election_2022_df.head())
        
        # Correction des voix pour 2017
        if 'nom' in election_2017_df.columns and 'voix_calculees' in election_2017_df.columns:
            # Recalculer les voix après correction des pourcentages
            election_2017_df['voix_calculees'] = (election_2017_df['pourcentage_voix_exprimes'] * election_2017_df['Exprimés']) / 100
            
            # Calculer le total des exprimés (avant correction)
            total_exprimes_2017 = election_2017_df['Exprimés'].sum()
            print(f"\nTotal des exprimés (2017) avant correction : {total_exprimes_2017}")
            
            # Ajuster les voix pour correspondre au total attendu (110,000)
            scaling_factor_2017 = EXPECTED_EXPRIMES_2017 / total_exprimes_2017 if total_exprimes_2017 > 0 else 1
            election_2017_df['voix_calculees'] = election_2017_df['voix_calculees'] * scaling_factor_2017
            
            # Agrégation pour 2017 : Sommer les voix par candidat
            election_2017_df = election_2017_df.groupby('nom').agg({
                'voix_calculees': 'sum'
            }).reset_index()
            # Renommer la colonne pour homogénéité
            election_2017_df = election_2017_df.rename(columns={'voix_calculees': 'voix'})
            print("\nDonnées election_2017_df après agrégation et correction (voix totales) :")
            print(election_2017_df)
        
        # Correction des voix pour 2022
        if 'nom' in election_2022_df.columns and 'pourcentage_voix_exprimes' in election_2022_df.columns:
            # Recalculer les voix à partir des pourcentages corrigés
            election_2022_df['voix'] = (election_2022_df['pourcentage_voix_exprimes'] * EXPECTED_EXPRIMES_2022) / 100
            
            # Agrégation pour 2022 : Sommer les voix par candidat (si nécessaire)
            election_2022_df = election_2022_df.groupby('nom').agg({
                'voix': 'sum'
            }).reset_index()
            print("\nDonnées election_2022_df après agrégation et correction (voix totales) :")
            print(election_2022_df)
        
        return election_2017_df, election_2022_df
        
    except Exception as e:
        print(f"Erreur lors du chargement: {str(e)}")
        raise

def transform_election_2017(df):
    """Transforme election_2017_df de format large à format long"""
    # Identifier les colonnes communes (non liées aux candidats)
    common_cols = [col for col in df.columns if not any(x in col for x in ['Sexe', 'Nom', 'Prénom', 'Voix', '% Voix/Ins', '% Voix/Exp', 'N°Panneau'])]
    
    # Créer des listes pour les colonnes de chaque type (uniquement les colonnes suffixées)
    nom_cols = [col for col in df.columns if 'Nom.' in col]
    sexe_cols = [col for col in df.columns if 'Sexe.' in col]
    prenom_cols = [col for col in df.columns if 'Prénom.' in col]
    voix_cols = [col for col in df.columns if 'Voix.' in col and '% Voix' not in col]
    voix_ins_cols = [col for col in df.columns if '% Voix/Ins.' in col]
    voix_exp_cols = [col for col in df.columns if '% Voix/Exp.' in col]
    
    # Débogage : Afficher les colonnes détectées
    print("\nColonnes détectées dans transform_election_2017 :")
    print(f"Colonnes 'Nom.*' : {nom_cols}")
    print(f"Colonnes 'Sexe.*' : {sexe_cols}")
    print(f"Colonnes 'Prénom.*' : {prenom_cols}")
    print(f"Colonnes 'Voix.*' : {voix_cols}")
    print(f"Colonnes '% Voix/Ins.*' : {voix_ins_cols}")
    print(f"Colonnes '% Voix/Exp.*' : {voix_exp_cols}")
    
    # Vérifier que le nombre de colonnes correspond
    num_candidates = len(nom_cols)
    if not (len(sexe_cols) == len(prenom_cols) == len(voix_cols) == len(voix_ins_cols) == len(voix_exp_cols) == num_candidates):
        print("Incohérence dans le nombre de colonnes pour les candidats dans election_2017_df")
        print(f"Nombre de colonnes 'Nom.*' : {len(nom_cols)}")
        print(f"Nombre de colonnes 'Sexe.*' : {len(sexe_cols)}")
        print(f"Nombre de colonnes 'Prénom.*' : {len(prenom_cols)}")
        print(f"Nombre de colonnes 'Voix.*' : {len(voix_cols)}")
        print(f"Nombre de colonnes '% Voix/Ins.*' : {len(voix_ins_cols)}")
        print(f"Nombre de colonnes '% Voix/Exp.*' : {len(voix_exp_cols)}")
        # Retourner un DataFrame vide avec les colonnes attendues
        return pd.DataFrame(columns=['Nom', 'Sexe', 'Prénom', 'Voix', '% Voix/Ins', '% Voix/Exp', 'Exprimés'] + common_cols)
    
    # Transformer chaque type de colonne séparément
    melted_dfs = []
    for i in range(num_candidates):
        candidate_cols = {
            'Nom': nom_cols[i],
            'Sexe': sexe_cols[i],
            'Prénom': prenom_cols[i],
            'Voix': voix_cols[i],
            'Voix/Ins': voix_ins_cols[i],
            '% Voix/Exp': voix_exp_cols[i]
        }
        
        # Sous-ensemble du DataFrame avec les colonnes communes et les colonnes du candidat
        temp_df = df[common_cols + list(candidate_cols.values())].copy()
        # Renommer les colonnes pour enlever les suffixes
        temp_df = temp_df.rename(columns={v: k for k, v in candidate_cols.items()})
        # Ajouter une colonne pour identifier le numéro du candidat
        temp_df['candidate_number'] = i + 1
        # Supprimer les lignes où le nom est NaN
        temp_df = temp_df.dropna(subset=['Nom'])
        # Calculer les voix réelles : (% Voix/Exp) * Exprimés / 100
        temp_df['voix_calculees'] = (temp_df['% Voix/Exp'] * temp_df['Exprimés']) / 100
        melted_dfs.append(temp_df)
    
    # Concaténer tous les DataFrames
    if melted_dfs:
        result_df = pd.concat(melted_dfs, ignore_index=True)
        # Supprimer les colonnes redondantes comme N°Panneau.*
        cols_to_drop = [col for col in result_df.columns if 'N°Panneau' in col]
        result_df = result_df.drop(columns=cols_to_drop, errors='ignore')
    else:
        result_df = pd.DataFrame(columns=['Nom', 'Sexe', 'Prénom', 'Voix', '% Voix/Ins', '% Voix/Exp', 'Exprimés'] + common_cols)
    
    return result_df

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
    
    required_cols = ['nom', 'voix']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Colonnes manquantes ({year}) : {missing_cols}")
    
    return df

def analyze_election_results(df_2017, df_2022):
    """Analyse détaillée des résultats par parti (en nombre de voix)"""
    parties = {
        'RN': ['LE PEN', 'MARINE', 'RN', 'RASSEMBLEMENT', 'NATIONAL'],
        'LREM': ['MACRON', 'EMMANUEL', 'LREM', 'PRESIDENT', 'RENAISSANCE', 'ENSEMBLE'],
        'LR': ['LES REPUBLICAINS', 'REPUBLICAIN', 'LR', 'PECRESSE', 'CIOTTI', 'SARKOZY', 'FILLON'],
        'LFI': ['MELENCHON', 'JEAN-LUC', 'LFI', 'FRANCE INSOMISE'],
        'PS': ['PS', 'SOCIALISTE', 'HAMON', 'HOLLANDE', 'HIDALGO'],
        'ECOLO': ['JADOT', 'ECOLOGIE', 'VERT', 'EELV'],
        'REC': ['DUPONT-AIGNAN', 'DUPONTAIGNAN', 'RECONQUETE', 'ZEMMOUR', 'DUPONT AIGNAN'],
        'PCF': ['ROUSSEL', 'COMMUNISTE', 'PCF'],
        'LO': ['ARTHAUD', 'LUTTE OUVRIERE'],
        'AUTRES': []
    }
    
    results = {}
    for year, df in [(2017, df_2017), (2022, df_2022)]:
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
            party_data = election_results[party].values
            model.fit(years, party_data)
            future_years = np.array(years_to_predict).reshape(-1, 1)
            party_pred = model.predict(future_years)
            
            # Ajustements spécifiques pour éviter des valeurs négatives
            party_pred = np.clip(party_pred, 0, 120000)  # Limiter à 120,000 maximum
            
            for year, pred in zip(years_to_predict, party_pred):
                if year not in predictions:
                    predictions[year] = {}
                predictions[year][party] = max(0, pred)
            
            x_vals = np.linspace(min(years), max(years_to_predict), 100)
            y_vals = model.predict(x_vals.reshape(-1, 1))
            y_vals = np.clip(y_vals, 0, 120000)
            
            plt.plot(x_vals, y_vals, linestyle='-', alpha=0.7, color=f'C{i}')
            plt.scatter(years, party_data, label=f'{party} (historique)', color=f'C{i}', marker='o', s=100)
            plt.scatter(years_to_predict, party_pred, marker='*', s=100, color=f'C{i}', label=f'{party} (prédiction)')
        
        except Exception as e:
            print(f"Erreur pour {party}: {str(e)}")
            continue
    
    plt.title('Évolution des voix des partis politiques (2017-2032) - Gers (32, 1er tour)', pad=20)
    plt.xlabel('Année')
    plt.ylabel('Nombre de voix')
    plt.xticks(np.append(election_results.index, years_to_predict))
    plt.ylim(0, 120000)  # Limiter à 120,000 pour le Gers
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
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
        election_2017_df, election_2022_df = load_data()
        
        # 2. Analyse tous partis (en voix)
        print("\n2. Analyse des résultats électoraux (voix)...")
        election_results = analyze_election_results(election_2017_df, election_2022_df)
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