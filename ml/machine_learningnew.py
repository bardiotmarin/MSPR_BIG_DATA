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

# --- 1. CHARGEMENT ET TRANSFORMATION WIDE -> LONG ---
def charger_et_transformer():
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

# --- 2. FEATURE ENGINEERING ---
def preparer_features(df):
    principaux = df["nom"].value_counts().nlargest(20).index
    df["candidat"] = df["nom"].where(df["nom"].isin(principaux), "AUTRE")
    df = df.sort_values(["code_region", "candidat", "annee"])
    df["voix_lag_5"] = df.groupby(["code_region", "candidat"])["voix"].shift(1)
    return df.dropna(subset=["voix_lag_5"])

# --- 3. ENTRAÎNEMENT ---
def entrainer_modele(df):
    features = ["annee", "voix_lag_5"]
    X = df[features]
    y = df["voix"]
    pipeline = make_pipeline(RobustScaler(), HistGradientBoostingRegressor(max_iter=300, random_state=42))
    pipeline.fit(X, y)
    # Évaluation du modèle sur les données d'entraînement
    y_pred = pipeline.predict(X)
    r2 = r2_score(y, y_pred)
    return pipeline, r2

# --- 4. PREDICTION ---
def predire_futur(df, modele, annees=[2027, 2032]):
    dernier = df.groupby(["code_region", "candidat"]).apply(lambda x: x.loc[x["annee"].idxmax()]).reset_index(drop=True)
    predictions = []
    for annee in annees:
        futur = dernier.copy()
        futur["annee"] = annee
        X_futur = futur[["annee", "voix_lag_5"]]
        futur["pred_voix"] = modele.predict(X_futur)
        futur["pred_voix"] = futur["pred_voix"].clip(0)
        futur["annee"] = annee
        predictions.append(futur)
    return pd.concat(predictions)

# --- 5. VISUALISATION ---
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
    df_long = charger_et_transformer()
    print("Feature engineering...")
    df_feat = preparer_features(df_long)
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