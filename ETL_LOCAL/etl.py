import pandas as pd

def load_data(file_paths):
    """Charge les fichiers Excel dans des DataFrames."""
    dataframes = [pd.read_excel(file) for file in file_paths]
    return dataframes

def clean_data(df):
    """Nettoie les données (suppression des NaN, correction des types, etc.)."""
    df = df.dropna()  # Supprime les lignes avec des valeurs manquantes
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")  # Normalise les noms de colonnes
    return df

def merge_data(dfs):
    """Fusionne les DataFrames sur une clé commune (ex: 'commune', 'département', 'année')."""
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

def save_data(df, output_file):
    """Enregistre le DataFrame final dans un fichier Excel."""
    df.to_excel(output_file, index=False)

def main():
    file_paths = ["C:\\Users\droui\Documents\\githubprojetcs\\MSPR_BIG_DATA\DATA\\raw\\donnee-reg-data.gouv-2024-geographie2024-produit-le2025-01-26.xlsx", "C:\\Users\\droui\Documents\\githubprojetcs\\MSPR_BIG_DATA\DATA\\raw\\Presidentielle_2017_Resultats_Tour_1.xls"]
    output_file = "elections_fusionnees.xlsx"
    
    # ETL Process
    dataframes = load_data(file_paths)
    cleaned_dataframes = [clean_data(df) for df in dataframes]
    merged_data = merge_data(cleaned_dataframes)
    save_data(merged_data, output_file)
    
    print(f"Fichier fusionné enregistré sous : {output_file}")

if __name__ == "__main__":
    main()
