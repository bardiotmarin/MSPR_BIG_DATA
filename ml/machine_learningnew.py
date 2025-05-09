import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- CONFIGURATION BASE DE DONNÉES ---
DATABASE_URL = "postgresql+psycopg2://user:password@localhost:5433/mspr_warehouse"
engine = create_engine(DATABASE_URL)

# --- 1. CHARGEMENT ET TRANSFORMATION WIDE -> LONG (ÉLECTIONS) ---
def charger_et_transformer_elections():
    query = "SELECT * FROM election_tour_1 WHERE annee BETWEEN 2002 AND 2022"
    df = pd.read_sql(query, engine)
    df["code_region"] = pd.to_numeric(df["code_region"], errors="coerce").astype("int64")
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
    print("Dimensions de df_elections après chargement:", df_long.shape)

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
    print("Partis présents dans df_elections:", sorted(df_long["parti"].unique()))
    return df_long

# --- 2. CHARGEMENT ET TRANSFORMATION CHOMAGE ---
def charger_et_transformer_chomage():
    query = "SELECT * FROM chomage_france"
    df = pd.read_sql(query, engine)
    trimestres = [f"{year}-T{trim}" for year in range(2002, 2023) for trim in range(1, 5)]
    df_long = pd.melt(df, var_name="trimestre", value_name="taux_chomage", value_vars=trimestres)
    df_long[["annee", "trimestre"]] = df_long["trimestre"].str.extract(r'(\d{4})-T(\d)')
    df_long["annee"] = df_long["annee"].astype(int)
    df_long["trimestre"] = df_long["trimestre"].astype(int)
    df_long = df_long.dropna(subset=["taux_chomage"])
    print("Dimensions de df_chomage après chargement:", df_long.shape)
    return df_long

# --- 3. CHARGEMENT ET TRANSFORMATION PAUVRETE ---
def charger_et_transformer_pauvrete():
    query = "SELECT * FROM pauvrete_france"
    df = pd.read_sql(query, engine)
    df = df.dropna(subset=["taux", "annee"])
    print("Dimensions de df_pauvrete après chargement:", df.shape)
    return df

# --- 4. CHARGEMENT ET TRANSFORMATION POLICE/GENDARMERIE ---
def charger_et_transformer_police():
    query = "SELECT annee, code_region, indicateur, taux_pour_mille FROM police_et_gendarmerie_statistique_france"
    df = pd.read_sql(query, engine)
    df["code_region"] = pd.to_numeric(df["code_region"], errors="coerce").astype("int64")
    df["taux_pour_mille"] = pd.to_numeric(df["taux_pour_mille"], errors="coerce")
    df_pivot = df.pivot_table(index=["annee", "code_region"], columns="indicateur", values="taux_pour_mille", aggfunc="mean").reset_index()
    df_pivot = df_pivot.rename_axis(None, axis=1)
    df_pivot = df_pivot.dropna()
    print("Dimensions de df_police après chargement et pivot:", df_pivot.shape)
    return df_pivot

# --- 5. FEATURE ENGINEERING ---
def preparer_features(df_elections, df_chomage, df_pauvrete, df_police):
    principaux = df_elections["nom"].value_counts().nlargest(20).index
    df_elections["candidat"] = df_elections["nom"].where(df_elections["nom"].isin(principaux), "AUTRE")
    df_elections = df_elections.sort_values(["code_region", "candidat", "annee"])
    df_elections["voix_lag_5"] = df_elections.groupby(["code_region", "candidat"])["voix"].shift(1)
    print("Dimensions après calcul de voix_lag_5:", df_elections.shape)
    print("Nombre de NaN dans voix_lag_5:", df_elections["voix_lag_5"].isna().sum())

    df_elections = df_elections.merge(df_chomage[["annee", "trimestre", "taux_chomage"]], on="annee", how="left")
    print("Dimensions après fusion avec chomage:", df_elections.shape)
    print("Nombre de NaN dans taux_chomage:", df_elections["taux_chomage"].isna().sum())

    df_elections = df_elections.merge(df_pauvrete[["annee", "taux"]], on="annee", how="left", suffixes=("", "_pauvrete"))
    print("Dimensions après fusion avec pauvrete:", df_elections.shape)
    print("Nombre de NaN dans taux:", df_elections["taux"].isna().sum())

    df_elections = df_elections.merge(df_police, on=["annee", "code_region"], how="left")
    print("Dimensions après fusion avec police:", df_elections.shape)

    df_elections["taux_chomage"] = df_elections["taux_chomage"].fillna(df_elections["taux_chomage"].mean())
    df_elections["taux"] = df_elections["taux"].fillna(df_elections["taux"].mean())
    police_columns = [col for col in df_elections.columns if col not in ["annee", "voix", "voix_lag_5", "taux_chomage", "taux", "code_region", "nom", "prenom", "sexe", "parti", "candidat", "trimestre"]]
    for col in police_columns:
        df_elections[col] = df_elections[col].fillna(df_elections[col].mean())
        print(f"Nombre de NaN dans {col}:", df_elections[col].isna().sum())

    df_elections = df_elections.dropna(subset=["voix_lag_5"])
    print("Dimensions après nettoyage final:", df_elections.shape)
    print("Partis présents après nettoyage:", sorted(df_elections["parti"].unique()))
    return df_elections

# --- 6. ENTRAÎNEMENT AVEC VALIDATION ---
def entrainer_modele(df):
    if df.empty:
        raise ValueError("Le DataFrame d'entraînement est vide après le prétraitement.")
    
    police_columns = [col for col in df.columns if col not in ["annee", "voix", "voix_lag_5", "taux_chomage", "taux", "code_region", "nom", "prenom", "sexe", "parti", "candidat", "trimestre"]]
    features = ["annee", "voix_lag_5", "taux_chomage", "taux"] + police_columns
    X = df[features]
    y = df["voix"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nNombre d'échantillons:")
    print(f"- Entraînement: {len(X_train)}")
    print(f"- Test: {len(X_test)}")
    
    pipeline = make_pipeline(
        RobustScaler(), 
        HistGradientBoostingRegressor(max_iter=300, random_state=42)
    )
    pipeline.fit(X_train, y_train)
    
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    print("\nPerformances du modèle:")
    print(f"- R² (entraînement): {r2_train:.3f}")
    print(f"- R² (test): {r2_test:.3f}")
    
    return pipeline, r2_train, r2_test

# --- 7. PREDICTION ---
def predire_futur(df, modele, annees=[2027, 2032]):
    all_candidates = df[["nom", "parti", "code_region"]].drop_duplicates()
    dernier = df.groupby(["code_region", "candidat"]).apply(lambda x: x.loc[x["annee"].idxmax()]).reset_index(drop=True)
    dernier = all_candidates.merge(dernier, on=["nom", "parti", "code_region"], how="left")
    dernier["voix_lag_5"] = dernier["voix_lag_5"].fillna(dernier.groupby("nom")["voix_lag_5"].transform("mean"))
    dernier["annee"] = dernier["annee"].fillna(df["annee"].max())
    
    police_columns = [col for col in df.columns if col not in ["annee", "voix", "voix_lag_5", "taux_chomage", "taux", "code_region", "nom", "prenom", "sexe", "parti", "candidat", "trimestre"]]
    for col in police_columns:
        if col not in dernier.columns:
            dernier[col] = df[df["annee"] == df["annee"].max()][col].mean()
        else:
            dernier[col] = dernier[col].fillna(df[df["annee"] == df["annee"].max()][col].mean())
    
    dernier_annee = df["annee"].max()
    taux_chomage_2027 = df[df["annee"] == dernier_annee]["taux_chomage"].mean()
    taux_pauvrete_2027 = df[df["annee"] == dernier_annee]["taux"].mean()
    police_values_2027 = {col: df[df["annee"] == dernier_annee][col].mean() for col in police_columns}
    
    taux_chomage_2032 = taux_chomage_2027 * 1.5
    taux_pauvrete_2032 = taux_pauvrete_2027 * 0.5
    police_values_2032 = {col: value * 2.0 for col, value in police_values_2027.items()}
    
    predictions = []
    for annee in annees:
        futur = dernier.copy()
        futur["annee"] = annee
        
        if annee == 2027:
            futur["taux_chomage"] = taux_chomage_2027
            futur["taux"] = taux_pauvrete_2027
            for col in police_columns:
                futur[col] = police_values_2027[col]
        elif annee == 2032:
            futur["taux_chomage"] = taux_chomage_2032
            futur["taux"] = taux_pauvrete_2032
            for col in police_columns:
                futur[col] = police_values_2032[col]
        
        X_futur = futur[["annee", "voix_lag_5", "taux_chomage", "taux"] + police_columns]
        futur["pred_voix"] = modele.predict(X_futur)
        futur["pred_voix"] = futur["pred_voix"].clip(0)
        futur["annee"] = annee
        predictions.append(futur)
    
    preds = pd.concat(predictions)
    print("Partis présents dans les prédictions:", sorted(preds["parti"].unique()))
    return preds

# --- 8. VISUALISATION ---
def plot_national(preds):
    all_partis = [
        "Rassemblement pour la République (RPR)", "Front National (FN)", "Parti Socialiste (PS)",
        "Chasse, Pêche, Nature et Traditions (CPNT)", "Lutte Ouvrière (LO)", "Parti Communiste Français (PCF)",
        "Parti Radical de Gauche (PRG)", "Union pour la Démocratie Française (UDF)", "Mouvement des Citoyens (MDC)",
        "Mouvement National Républicain (MNR)", "Cap 21", "Démocratie Libérale (DL)", "Forum des Républicains Sociaux (FRS)",
        "Parti des Travailleurs (PT)", "Les Verts", "Union pour un Mouvement Populaire (UMP)", "Ligue Communiste Révolutionnaire (LCR)",
        "Mouvement pour la France (MPF)", "Front de Gauche (FG)", "Europe Écologie Les Verts (EELV)", "Debout la République (DLR)",
        "Nouveau Parti Anticapitaliste (NPA)", "Solidarité et Progrès (S&P)", "En Marche ! (EM)", "Les Républicains (LR)",
        "Résistons !", "Union Populaire Républicaine (UPR)", "Reconquête", "AUTRE"
    ]
    
    # Définir une palette de couleurs distinctes (au moins 30 couleurs)
    couleurs = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", 
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", 
        "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5", "#ff6f61", "#6b5b95", "#feb236", "#d64161", 
        "#88b04b", "#92a8d1", "#f4a261", "#e59866", "#48c9b0", "#af7ac5", "#5499c7", "#82e0aa"
    ]
    
    # Associer chaque parti à une couleur
    parti_colors = {parti: couleurs[i % len(couleurs)] for i, parti in enumerate(all_partis)}
    
    national = preds.groupby(["annee", "parti"])["pred_voix"].sum().unstack()
    
    for parti in all_partis:
        if parti not in national.columns:
            national[parti] = 0.0
    
    # Plot avec couleurs personnalisées
    national.plot(
        marker='o', 
        figsize=(12, 8),
        color=[parti_colors[parti] for parti in national.columns]
    )
    plt.title("Tendance élection présidentielle pour 2027/2032 avec indicateurs socio-économiques/policiers")
    plt.ylabel("Nombre total de voix")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

# --- MAIN ---
if __name__ == "__main__":
    print("Chargement et transformation des données...")
    df_elections = charger_et_transformer_elections()
    df_chomage = charger_et_transformer_chomage()
    df_pauvrete = charger_et_transformer_pauvrete()
    df_police = charger_et_transformer_police()
    
    print("\nFeature engineering...")
    df_feat = preparer_features(df_elections, df_chomage, df_pauvrete, df_police)
    
    print("\nEntraînement du modèle avec validation...")
    modele, r2_train, r2_test = entrainer_modele(df_feat)
    
    print("\nPrédictions pour 2027 et 2032...")
    preds = predire_futur(df_feat, modele, [2027, 2032])
    preds.to_csv("predictions_tour1_2027_2032.csv", index=False)
    
    print("\nVisualisation nationale...")
    plot_national(preds)
    
    print("\nRésultats:")
    print(f"- Fichier de prédictions sauvegardé: predictions_tour1_2027_2032.csv")
    print(f"- Performance modèle (R²):")
    print(f"  • Entraînement: {r2_train:.3f}")
    print(f"  • Test: {r2_test:.3f}")
    print("\nNote : L'écart entre R² train et test indique le degré de surapprentissage.")