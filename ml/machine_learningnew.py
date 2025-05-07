import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# --- CONFIGURATION BASE DE DONNÉES ---
DATABASE_URL = "postgresql+psycopg2://user:password@localhost:5433/mspr_warehouse"
engine = create_engine(DATABASE_URL)

# --- 1. CHARGEMENT ET TRANSFORMATION WIDE -> LONG (ÉLECTIONS) ---
def charger_et_transformer_elections():
    query = "SELECT * FROM election_tour_1"
    df = pd.read_sql(query, engine)
    candidats = []
    for i in range(1, 17):
        suf = "" if i == 1 else f"_{i}"
        candidats.append(pd.DataFrame({
            "annee": df["annee"],
            "code_region": df["code_region"],
            "nom": df[f"nom{suf}"],
            "prenom": df[f"prenom{suf}"],
            "sexe": df[f"sexe{suf}"],
            "voix": df[f"voix{suf}"]
        }))
    df_long = pd.concat(candidats, ignore_index=True)
    df_long = df_long.dropna(subset=["nom", "voix"])

    # Mapping statique des candidats aux partis (2002-2022)
    parti_mapping = {
        "CHIRAC": "Rassemblement pour la République (RPR)",
        "LE PEN": "Front National (FN)",
        "JOSPIN": "Parti Socialiste (PS)",
        "SAINT-JOSSE": "Chasse, Pêche, Nature et Traditions (CPNT)",
        "LAGUILLER": "Lutte Ouvrière (LO)",
        "HUE": "Parti Communiste Français (PCF)",
        "TAUBIRA": "Parti Radical de Gauche (PRG)",
        "BAYROU": "Union pour la Démocratie Française (UDF)",
        "CHEVENEMENT": "Mouvement des Citoyens (MDC)",
        "MEGRET": "Mouvement National Républicain (MNR)",
        "LEPAGE": "Cap 21",
        "MADELIN": "Démocratie Libérale (DL)",
        "BOUTIN": "Forum des Républicains Sociaux (FRS)",
        "GLUCKSTEIN": "Parti des Travailleurs (PT)",
        "MAMERE": "Les Verts",
        "SARKOZY": "Union pour un Mouvement Populaire (UMP)",
        "ROYAL": "Parti Socialiste (PS)",
        "BESANCENOT": "Ligue Communiste Révolutionnaire (LCR)",
        "BUFFET": "Parti Communiste Français (PCF)",
        "VOYNET": "Les Verts",
        "DE VILLIERS": "Mouvement pour la France (MPF)",
        "NIHOUS": "Chasse, Pêche, Nature et Traditions (CPNT)",
        "SCHIVARDI": "Parti des Travailleurs (PT)",
        "HOLLANDE": "Parti Socialiste (PS)",
        "MELENCHON": "Front de Gauche (FG)",
        "JOLY": "Europe Écologie Les Verts (EELV)",
        "DUPONT-AIGNAN": "Debout la République (DLR)",
        "POUTOU": "Nouveau Parti Anticapitaliste (NPA)",
        "ARTHAUD": "Lutte Ouvrière (LO)",
        "CHEMINADE": "Solidarité et Progrès (S&P)",
        "MACRON": "En Marche ! (EM)",
        "FILLON": "Les Républicains (LR)",
        "HAMON": "Parti Socialiste (PS)",
        "LASSALLE": "Résistons !",
        "ASSELINEAU": "Union Populaire Républicaine (UPR)",
        "ZEMMOUR": "Reconquête",
        "PECRESSSE": "Les Républicains (LR)",
        "JADOT": "Europe Écologie Les Verts (EELV)",
        "HIDALGO": "Parti Socialiste (PS)",
        "ROUSSEL": "Parti Communiste Français (PCF)"
    }
    df_long["parti"] = df_long["nom"].map(parti_mapping).fillna("AUTRE")
    return df_long

# --- 2. CHARGEMENT ET TRANSFORMATION CHOMAGE ---
def charger_et_transformer_chomage():
    query = "SELECT * FROM chomage_france"
    df = pd.read_sql(query, engine)
    # Créer une liste de colonnes de trimestre (2002-T1 à 2022-T4)
    trimestres = [f"{year}-T{trim}" for year in range(2002, 2023) for trim in range(1, 5)]
    # Transformer en format long
    df_long = pd.melt(df, var_name="trimestre", value_name="taux_chomage", value_vars=trimestres)
    # Extraire année et trimestre
    df_long[["annee", "trimestre"]] = df_long["trimestre"].str.extract(r'(\d{4})-T(\d)')
    df_long["annee"] = df_long["annee"].astype(int)
    df_long["trimestre"] = df_long["trimestre"].astype(int)
    df_long = df_long.dropna(subset=["taux_chomage"])
    return df_long

# --- 3. CHARGEMENT ET TRANSFORMATION PAUVRETE ---
def charger_et_transformer_pauvrete():
    query = "SELECT * FROM pauvrete_france"
    df = pd.read_sql(query, engine)
    df = df.dropna(subset=["taux", "annee"])
    return df

# --- 4. FEATURE ENGINEERING ---
def preparer_features(df_elections, df_chomage, df_pauvrete):
    principaux = df_elections["nom"].value_counts().nlargest(20).index
    df_elections["candidat"] = df_elections["nom"].where(df_elections["nom"].isin(principaux), "AUTRE")
    df_elections = df_elections.sort_values(["code_region", "candidat", "annee"])
    df_elections["voix_lag_5"] = df_elections.groupby(["code_region", "candidat"])["voix"].shift(1)

    # Fusionner avec chomage et pauvrete par annee
    df_elections = df_elections.merge(df_chomage[["annee", "trimestre", "taux_chomage"]], on="annee", how="left")
    df_elections = df_elections.merge(df_pauvrete[["annee", "taux"]], on="annee", how="left", suffixes=("", "_pauvrete"))

    return df_elections.dropna(subset=["voix_lag_5", "taux_chomage", "taux"])

# --- 5. ENTRAÎNEMENT ---
def entrainer_modele(df):
    features = ["annee", "voix_lag_5", "taux_chomage", "taux"]
    X = df[features]
    y = df["voix"]
    pipeline = make_pipeline(RobustScaler(), HistGradientBoostingRegressor(max_iter=300, random_state=42))
    pipeline.fit(X, y)
    # Évaluation du modèle sur les données d'entraînement
    y_pred = pipeline.predict(X)
    r2 = r2_score(y, y_pred)
    return pipeline, r2

# --- 6. PREDICTION ---
def predire_futur(df, modele, annees=[2027, 2032]):
    dernier = df.groupby(["code_region", "candidat"]).apply(lambda x: x.loc[x["annee"].idxmax()]).reset_index(drop=True)
    predictions = []
    for annee in annees:
        futur = dernier.copy()
        futur["annee"] = annee
        # Remplir taux_chomage et taux avec les dernières valeurs connues (approximation)
        dernier_annee = df["annee"].max()
        futur["taux_chomage"] = df[df["annee"] == dernier_annee]["taux_chomage"].mean()
        futur["taux"] = df[df["annee"] == dernier_annee]["taux"].mean()
        X_futur = futur[["annee", "voix_lag_5", "taux_chomage", "taux"]]
        futur["pred_voix"] = modele.predict(X_futur)
        futur["pred_voix"] = futur["pred_voix"].clip(0)
        futur["annee"] = annee
        predictions.append(futur)
    return pd.concat(predictions)

# --- 7. VISUALISATION ---
def plot_national(preds):
    national = preds.groupby(["annee", "parti"])["pred_voix"].sum().unstack()
    national.plot(marker='o', figsize=(12, 6))
    plt.title("Projection nationale premier tour (2027/2032) par parti")
    plt.ylabel("Nombre total de voix")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    print("Chargement et transformation des données...")
    df_elections = charger_et_transformer_elections()
    df_chomage = charger_et_transformer_chomage()
    df_pauvrete = charger_et_transformer_pauvrete()
    print("Feature engineering...")
    df_feat = preparer_features(df_elections, df_chomage, df_pauvrete)
    print("Entraînement du modèle...")
    modele, r2_score_value = entrainer_modele(df_feat)
    print("Prédictions pour 2027 et 2032...")
    preds = predire_futur(df_feat, modele, [2027, 2032])
    preds.to_csv("predictions_tour1_2027_2032.csv", index=False)
    print("Visualisation nationale...")
    plot_national(preds)
    print(f"Fini ! Résultats dans predictions_tour1_2027_2032.csv")
    print(f"Score de fiabilité (R²) : {r2_score_value:.2f}")
    print("Note : Ce score est basé sur les données d'entraînement. Une validation croisée serait plus robuste.")