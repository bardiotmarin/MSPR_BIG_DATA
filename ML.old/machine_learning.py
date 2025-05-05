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
    """Charge les données électorales depuis PostgreSQL"""
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
        
        # Standardisation des colonnes
        election_2017_df = standardize_columns(election_2017_df, year=2017)
        election_2022_df = standardize_columns(election_2022_df, year=2022)
        
        # Débogage : Afficher les DataFrames après standardisation
        print("\nDonnées election_2017_df après standardisation :")
        if 'nom' in election_2017_df.columns and 'pourcentage_voix_exprimes' in election_2017_df.columns:
            print(election_2017_df[['nom', 'pourcentage_voix_exprimes']].head())
        else:
            print("Colonnes 'nom' ou 'pourcentage_voix_exprimes' manquantes dans election_2017_df")
            print(election_2017_df.head())
        
        print("\nDonnées election_2022_df après standardisation :")
        if 'nom' in election_2022_df.columns and 'pourcentage_voix_exprimes' in election_2022_df.columns:
            print(election_2022_df[['nom', 'pourcentage_voix_exprimes']].head())
        else:
            print("Colonnes 'nom' ou 'pourcentage_voix_exprimes' manquantes dans election_2022_df")
            print(election_2022_df.head())
        
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
        return pd.DataFrame(columns=['Nom', 'Sexe', 'Prénom', 'Voix', '% Voix/Ins', '% Voix/Exp'] + common_cols)
    
    # Transformer chaque type de colonne séparément
    melted_dfs = []
    for i in range(num_candidates):
        candidate_cols = {
            'Nom': nom_cols[i],
            'Sexe': sexe_cols[i],
            'Prénom': prenom_cols[i],
            'Voix': voix_cols[i],
            '% Voix/Ins': voix_ins_cols[i],
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
        melted_dfs.append(temp_df)
    
    # Concaténer tous les DataFrames
    if melted_dfs:
        result_df = pd.concat(melted_dfs, ignore_index=True)
        # Supprimer les colonnes redondantes comme N°Panneau.*
        cols_to_drop = [col for col in result_df.columns if 'N°Panneau' in col]
        result_df = result_df.drop(columns=cols_to_drop, errors='ignore')
    else:
        result_df = pd.DataFrame(columns=['Nom', 'Sexe', 'Prénom', 'Voix', '% Voix/Ins', '% Voix/Exp'] + common_cols)
    
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
            '% Voix/Exp': 'pourcentage_voix_exprimes'
        }
    }
    
    df = df.rename(columns={k: v for k, v in column_mapping[year].items() if k in df.columns})
    
    # Vérifier si la colonne 'nom' existe avant de normaliser
    if 'nom' in df.columns:
        # Vérifier si 'nom' est unique (pas de colonnes dupliquées)
        nom_cols = [col for col in df.columns if col == 'nom']
        if len(nom_cols) > 1:
            print(f"Erreur : Plusieurs colonnes 'nom' détectées dans le DataFrame pour l'année {year}")
            # Garder la dernière colonne 'nom' (celle créée par transform_election_2017)
            df = df.loc[:, ~df.columns.duplicated(keep='last')]
        
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
    
    required_cols = ['nom', 'pourcentage_voix_exprimes']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"Colonnes manquantes ({year}) : {missing_cols}")
    
    return df

def analyze_election_results(df_2017, df_2022):
    """Analyse détaillée des résultats par parti"""
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
        if 'pourcentage_voix_exprimes' not in df.columns:
            print(f"Avertissement : Colonne 'pourcentage_voix_exprimes' manquante pour l'année {year}")
            total_votes = 0
        else:
            total_votes = df['pourcentage_voix_exprimes'].sum()  # Total des pourcentages exprimés
        
        # Vérification : S'assurer que la somme est proche de 100%
        if not 95 <= total_votes <= 105:
            print(f"Erreur : La somme des pourcentages exprimés pour {year} est {total_votes:.2f}%, ce qui est incohérent. Vérifiez les données dans la base.")
        
        # Débogage : Afficher tous les noms dans le DataFrame
        if 'nom' in df.columns:
            print(f"\nNoms des candidats ({year}) :")
            print(df['nom'].unique())
        else:
            print(f"Avertissement : Colonne 'nom' manquante pour l'année {year}")
        
        # Vérifier la somme des pourcentages exprimés
        print(f"Total des pourcentages exprimés ({year}) : {total_votes:.2f}%")
        
        assigned_votes = 0
        assigned_candidates = []
        for party, keywords in parties.items():
            if keywords:
                if 'nom' not in df.columns:
                    year_results[party] = 0
                    continue
                # S'assurer que la colonne nom est de type string
                df['nom'] = df['nom'].astype(str).fillna('')
                mask = df['nom'].str.contains('|'.join(keywords), case=False, na=False)
                votes = df[mask]['pourcentage_voix_exprimes'].sum() if 'pourcentage_voix_exprimes' in df.columns else 0
                year_results[party] = votes
                assigned_votes += votes
                # Ajouter les candidats détectés
                assigned_candidates.extend(df[mask]['nom'].unique())
            else:
                year_results[party] = 0
        
        # Calculer les votes non attribués (AUTRES)
        year_results['AUTRES'] = max(0, total_votes - assigned_votes)
        
        # Débogage : Afficher les candidats non attribués (ceux dans AUTRES)
        if 'nom' in df.columns:
            unassigned_candidates = df[~df['nom'].isin(assigned_candidates)]['nom'].unique()
            print(f"\nCandidats non attribués (comptés dans AUTRES) pour {year} :")
            print(unassigned_candidates)
        
        # Normalisation pour que la somme soit 100%
        if total_votes > 0:
            for party in year_results:
                year_results[party] = (year_results[party] / total_votes) * 100
        else:
            print(f"Avertissement: Aucun vote valide pour l'année {year}")
        
        # Débogage : Afficher les résultats avant et après normalisation
        print(f"\nRésultats bruts avant normalisation ({year}) :")
        print({k: v for k, v in year_results.items()})
        
        # Vérifier la somme après normalisation
        total_normalized = sum(year_results.values())
        print(f"Somme des pourcentages après normalisation ({year}) : {total_normalized:.2f}%")
        
        # Réajuster pour que la somme soit exactement 100%
        if total_normalized != 100:
            factor = 100 / total_normalized
            for party in year_results:
                year_results[party] *= factor
        
        results[year] = year_results
    
    return pd.DataFrame(results).T

def predict_party_popularity(election_results, years_to_predict=[2027, 2032]):
    """Prédiction de popularité des partis avec ajustements pour le Gers"""
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
            party_data = election_results[party].values
            model.fit(years, party_data)
            future_years = np.array(years_to_predict).reshape(-1, 1)
            party_pred = model.predict(future_years)
            
            # Ajustements spécifiques pour le Gers (plus dynamiques)
            last_value = party_data[-1]
            if party == 'RN':
                # Limiter la croissance du RN à 60% max en 2032
                party_pred = np.clip(party_pred, 0, 60)
            elif party == 'LFI':
                # Assouplir encore plus pour LFI
                party_pred = np.clip(party_pred, 0, 50)
            elif party == 'PS':
                # Permettre une remontée du PS jusqu'à 20%
                party_pred = np.clip(party_pred, 0, 20)
            else:
                # Limiter les variations extrêmes pour les autres partis
                party_pred = np.clip(party_pred, 0, 50)
            
            for year, pred in zip(years_to_predict, party_pred):
                if year not in predictions:
                    predictions[year] = {}
                predictions[year][party] = max(1, pred)
            
            x_vals = np.linspace(min(years), max(years_to_predict), 100)
            y_vals = model.predict(x_vals.reshape(-1, 1))
            # Appliquer des limites moins strictes aux courbes
            if party == 'RN':
                y_vals = np.clip(y_vals, 0, 60)
            elif party == 'LFI':
                y_vals = np.clip(y_vals, 0, 50)
            elif party == 'PS':
                y_vals = np.clip(y_vals, 0, 20)
            else:
                y_vals = np.clip(y_vals, 0, 50)
            
            plt.plot(x_vals, y_vals, linestyle='-', alpha=0.7, color=f'C{i}')
            plt.scatter(years, party_data, label=f'{party} (historique)', color=f'C{i}', marker='o', s=100)
            plt.scatter(years_to_predict, party_pred, marker='*', s=100, color=f'C{i}', label=f'{party} (prédiction)')
        
        except Exception as e:
            print(f"Erreur pour {party}: {str(e)}")
            continue
    
    # Normalisation des prédictions pour que la somme soit 100%
    for year in predictions:
        total = sum(predictions[year].values())
        if total > 0:
            for party in predictions[year]:
                predictions[year][party] = (predictions[year][party] / total) * 100
        if total < 100:
            predictions[year]['AUTRES'] = 100 - sum(predictions[year].values())
        else:
            predictions[year]['AUTRES'] = 0
    
    plt.title('Prédiction de popularité des partis politiques (2017-2032) - Gers (32, 1er tour)', pad=20)
    plt.xlabel('Année')
    plt.ylabel('Part des votes (%)')
    plt.xticks(np.append(election_results.index, years_to_predict))
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n🔮 Prédictions de popularité par parti 🔮")
    for year in predictions:
        sorted_parties = sorted(predictions[year].items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 {year} - Parti prédominant: {sorted_parties[0][0]} ({sorted_parties[0][1]:.1f}%)")
        for party, score in sorted_parties:
            print(f"  - {party}: {score:.1f}%")
    
    return predictions

def main():
    print("=== ANALYSE ET PRÉDICTIONS DES ÉLECTIONS PRÉSIDENTIELLES - GERS (32, 1er tour) ===")
    
    try:
        # 1. Chargement des données
        print("\n1. Chargement des données...")
        election_2017_df, election_2022_df = load_data()
        
        # 2. Analyse tous partis
        print("\n2. Analyse des résultats électoraux...")
        election_results = analyze_election_results(election_2017_df, election_2022_df)
        print("\nRésultats électoraux normalisés :")
        print(election_results)
        
        # 3. Prédictions
        print("\n3. Préparation des prédictions...")
        predictions = predict_party_popularity(election_results)
        
    except Exception as e:
        print(f"\nERREUR: {str(e)}")
    
    print("\n=== ANALYSE TERMINÉE ===")

if __name__ == "__main__":
    main()