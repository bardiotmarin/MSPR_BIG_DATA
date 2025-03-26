import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Charger les données
file_path = "C:\\Users\\droui\\Documents\\githubprojetcs\\MSPR_BIG_DATA\\DATA\\raw\\fichierdelinquancepresidence.xlsx"
df = pd.read_excel(file_path)

# Nettoyage des données
df = df.dropna().drop_duplicates()

# Assurez-vous que la colonne "annee" est bien un entier
df['annee'] = df['annee'].astype(int)

# Identifier l'année la plus récente et ses valeurs
last_year_data = df[df['annee'] == df['annee'].max()]
latest_taux = last_year_data['taux_pour_mille'].values[0]
latest_nombre = last_year_data['nombre'].values[0]

# Calculer la tendance annuelle moyenne
df['taux_pour_mille_diff'] = df['taux_pour_mille'].diff().fillna(0)
df['nombre_diff'] = df['nombre'].diff().fillna(0)

avg_taux_change = df['taux_pour_mille_diff'].mean()
avg_nombre_change = df['nombre_diff'].mean()

# Estimation des valeurs pour 2027
years_to_predict = 2027 - df['annee'].max()
estimated_taux_2027 = latest_taux + avg_taux_change * years_to_predict
estimated_nombre_2027 = latest_nombre + avg_nombre_change * years_to_predict

print(f"Estimation pour 2027 - taux_pour_mille: {estimated_taux_2027}, nombre: {estimated_nombre_2027}")

# Création du dataset pour le modèle
features = ['taux_pour_mille', 'nombre']
target = 'Voix'

X = df[features]
y = df[target]

# Séparation des données en entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prédiction pour 2027
valeurs_2027 = np.array([[estimated_taux_2027, estimated_nombre_2027]])
prediction_2027 = model.predict(valeurs_2027)

print(f"Prédiction des voix pour 2027 : {prediction_2027[0]}")

# ---- VISUALISATION ----
# Création du graphique
plt.figure(figsize=(10, 5))

# Tracer l'évolution du nombre de crimes par année
plt.plot(df["annee"], df["nombre"], marker="o", linestyle="-", color="red", label="Nombre de crimes")

# Mise en forme
plt.xlabel("Année")
plt.ylabel("Nombre de crimes")
plt.title("Évolution du nombre de crimes par an")
plt.legend()
plt.grid(True)

# Affichage
plt.show()
