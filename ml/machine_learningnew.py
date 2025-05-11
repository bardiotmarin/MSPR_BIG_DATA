import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, accuracy_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Mapping des candidats aux catégories politiques
    categorie_mapping = {
        "CHIRAC": "Droite", "LE PEN": "Extrême droite", "JOSPIN": "Gauche", "SAINT-JOSSE": "Extrême droite",
        "LAGUILLER": "Extrême gauche", "HUE": "Gauche", "TAUBIRA": "Gauche", "BAYROU": "Centre",
        "CHEVENEMENT": "Centre", "MEGRET": "Extrême droite", "LEPAGE": "Centre", "MADELIN": "Droite",
        "BOUTIN": "Droite", "GLUCKSTEIN": "Extrême gauche", "MAMERE": "Gauche", "SARKOZY": "Droite",
        "ROYAL": "Gauche", "BESANCENOT": "Extrême gauche", "BUFFET": "Gauche", "VOYNET": "Gauche",
        "DE VILLIERS": "Droite", "NIHOUS": "Extrême droite", "SCHIVARDI": "Extrême gauche",
        "HOLLANDE": "Gauche", "MELENCHON": "Gauche", "JOLY": "Gauche", "DUPONT-AIGNAN": "Droite",
        "POUTOU": "Extrême gauche", "ARTHAUD": "Extrême gauche", "CHEMINADE": "Extrême gauche",
        "MACRON": "Centre", "FILLON": "Droite", "HAMON": "Gauche", "LASSALLE": "Centre",
        "ASSELINEAU": "Centre", "ZEMMOUR": "Extrême droite", "PECRESSSE": "Droite", "JADOT": "Gauche",
        "HIDALGO": "Gauche", "ROUSSEL": "Gauche"
    }
    df_long["categorie"] = df_long["nom"].map(categorie_mapping).fillna("Autres")
    print("Catégories présentes dans df_elections:", sorted(df_long["categorie"].unique()))
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
    police_columns = [col for col in df_elections.columns if col not in ["annee", "voix", "voix_lag_5", "taux_chomage", "taux", "code_region", "nom", "prenom", "sexe", "categorie", "candidat", "trimestre"]]
    for col in police_columns:
        df_elections[col] = df_elections[col].fillna(df_elections[col].mean())
        print(f"Nombre de NaN dans {col}:", df_elections[col].isna().sum())

    df_elections = df_elections.dropna(subset=["voix_lag_5"])
    print("Dimensions après nettoyage final:", df_elections.shape)
    print("Catégories présentes après nettoyage:", sorted(df_elections["categorie"].unique()))
    return df_elections

# --- 6. ENTRAÎNEMENT AVEC VALIDATION ---
def entrainer_modele(df):
    if df.empty:
        raise ValueError("Le DataFrame d'entraînement est vide après le prétraitement.")
    
    police_columns = [col for col in df.columns if col not in ["annee", "voix", "voix_lag_5", "taux_chomage", "taux", "code_region", "nom", "prenom", "sexe", "categorie", "candidat", "trimestre"]]
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
    
    print("\nPerformances du modèle (régression):")
    print(f"- R² (entraînement): {r2_train:.3f}")
    print(f"- R² (test): {r2_test:.3f}")
    
    return pipeline, r2_train, r2_test, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test

# --- 7. PREDICTION ---
def predire_futur(df, modele, annees=[2027, 2032]):
    all_candidates = df[["nom", "categorie", "code_region"]].drop_duplicates()
    dernier = df.groupby(["code_region", "candidat"]).apply(lambda x: x.loc[x["annee"].idxmax()]).reset_index(drop=True)
    dernier = all_candidates.merge(dernier, on=["nom", "categorie", "code_region"], how="left")
    dernier["voix_lag_5"] = dernier["voix_lag_5"].fillna(dernier.groupby("nom")["voix_lag_5"].transform("mean"))
    dernier["annee"] = dernier["annee"].fillna(df["annee"].max())
    
    police_columns = [col for col in df.columns if col not in ["annee", "voix", "voix_lag_5", "taux_chomage", "taux", "code_region", "nom", "prenom", "sexe", "categorie", "candidat", "trimestre"]]
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
    print("Catégories présentes dans les prédictions:", sorted(preds["categorie"].unique()))
    return preds

# --- 8. VISUALISATION NATIONALE ---
def plot_national(preds):
    all_categories = ["Extrême gauche", "Gauche", "Centre", "Droite", "Extrême droite", "Autres"]
    
    # Définir une palette de couleurs distinctes pour les catégories
    couleurs = {
        "Extrême gauche": "#ff6f61",
        "Gauche": "#2ca02c",
        "Centre": "#ffbb78",
        "Droite": "#1f77b4",
        "Extrême droite": "#d62728",
        "Autres": "#7f7f7f"
    }
    
    national = preds.groupby(["annee", "categorie"])["pred_voix"].sum().unstack()
    
    for categorie in all_categories:
        if categorie not in national.columns:
            national[categorie] = 0.0
    
    # Plot avec couleurs personnalisées
    national.plot(
        marker='o', 
        figsize=(12, 8),
        color=[couleurs[categorie] for categorie in national.columns]
    )
    plt.title("Tendance élection présidentielle pour 2027/2032 avec indicateurs socio-économiques/policiers")
    plt.ylabel("Nombre total de voix")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.show()

# --- 9. VISUALISATION : R² ET MÉTRIQUES DE CLASSIFICATION ---
def plot_regression_and_classification_metrics(df, X_test, y_test, y_pred_test, r2_test):
    """
    Visualise le R² score (régression) et les métriques de classification (accuracy, précision, F1 score)
    pour la prédiction de la catégorie majoritaire.
    """
    # --- Étape 1 : Transformer les prédictions de régression en classification ---
    # Associer les prédictions à leurs catégories et régions
    df_test = df.loc[X_test.index].copy()
    df_test["pred_voix"] = y_pred_test
    df_test["voix"] = y_test

    # Identifier la catégorie majoritaire (réelle et prédite) par région et année
    df_test_grouped = df_test.groupby(["annee", "code_region", "categorie"]).agg({
        "voix": "sum",
        "pred_voix": "sum"
    }).reset_index()

    # Catégorie majoritaire réelle
    idx_true = df_test_grouped.groupby(["annee", "code_region"])["voix"].idxmax()
    true_majoritaire = df_test_grouped.loc[idx_true, ["annee", "code_region", "categorie"]]
    true_majoritaire = df_test_grouped.loc[idx_true, ["annee", "code_region", "categorie"]]
    true_majoritaire = true_majoritaire.rename(columns={"categorie": "categorie_true"})

    # Catégorie majoritaire prédite
    idx_pred = df_test_grouped.groupby(["annee", "code_region"])["pred_voix"].idxmax()
    pred_majoritaire = df_test_grouped.loc[idx_pred, ["annee", "code_region", "categorie"]]
    pred_majoritaire = pred_majoritaire.rename(columns={"categorie": "categorie_pred"})

    # Fusionner les résultats
    classification_results = true_majoritaire.merge(
        pred_majoritaire, 
        on=["annee", "code_region"], 
        how="inner"
    )

    # --- Étape 2 : Calculer les métriques de classification ---
    y_true = classification_results["categorie_true"]
    y_pred = classification_results["categorie_pred"]

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\nPerformances de classification (catégorie majoritaire):")
    print(f"- Accuracy: {accuracy:.3f}")
    print(f"- Précision: {precision:.3f}")
    print(f"- F1 Score: {f1:.3f}")

    # --- Étape 3 : Visualisation ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Graphique 1 : R² (dispersion des valeurs réelles vs prédites)
    ax1.scatter(y_test, y_pred_test, alpha=0.5, color="blue", label="Prédictions")
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ligne parfaite")
    ax1.set_xlabel("Voix réelles")
    ax1.set_ylabel("Voix prédites")
    ax1.set_title(f"R² = {r2_test:.3f}")
    ax1.legend()
    ax1.grid(True)

    # Graphique 2 : Métriques de classification
    metrics = {"Accuracy": accuracy, "Précision": precision, "F1 Score": f1}
    bars = ax2.bar(metrics.keys(), metrics.values(), color=["#ff6f61", "#2ca02c", "#ffbb78"])
    ax2.set_ylim(0, 1)
    ax2.set_title("Métriques de classification (catégorie majoritaire)")
    ax2.set_ylabel("Score")
    ax2.grid(True, axis="y")

    # Ajouter les valeurs sur les barres
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom")

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
    modele, r2_train, r2_test, X_train, X_test, y_train, y_test, y_pred_train, y_pred_test = entrainer_modele(df_feat)
    
    print("\nPrédictions pour 2027 et 2032...")
    preds = predire_futur(df_feat, modele, [2027, 2032])
    preds.to_csv("predictions_tour1_2027_2032.csv", index=False)
    
    print("\nVisualisation nationale...")
    plot_national(preds)
    
    print("\nVisualisation du R² et des métriques de classification...")
    plot_regression_and_classification_metrics(df_feat, X_test, y_test, y_pred_test, r2_test)
    
    print("\nRésultats:")
    print(f"- Fichier de prédictions sauvegardé: predictions_tour1_2027_2032.csv")
    print(f"- Performance modèle (R²):")
    print(f"  • Entraînement: {r2_train:.3f}")
    print(f"  • Test: {r2_test:.3f}")
    print("\nNote : L'écart entre R² train et test indique le degré de surapprentissage.")