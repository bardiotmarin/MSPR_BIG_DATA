import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 1. Chargement des données
# Remplace 'donnees.csv' par le nom de ton fichier contenant les données consolidées
df = pd.read_csv("donnees.csv")

# 2. Vérification des données
print(df.head())
print(df.info())
print(df.isnull().sum())  # Vérifier les valeurs manquantes

df = df.fillna(df.mean())  # Imputation des valeurs manquantes si nécessaire

# 3. Définition des variables
X = df.drop(columns=["Vote_Candidat_A", "Année"])  # On exclut la variable cible et l'année
y = df["Vote_Candidat_A"]

# 4. Séparation des données en entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Entraînement du modèle Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Prédictions et évaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Erreur moyenne absolue : {mae:.2f}")

# 7. Prédiction pour l'année 2027 (exemple)
nouvelle_donnee = pd.DataFrame({
    "Délinquance": [1400],
    "Crimes": [350],
    "Chômage": [7.2]
})

prediction = model.predict(nouvelle_donnee)
print(f"Prédiction du vote pour 2027 : {prediction[0]:.2f}%")

# 8. Visualisation des résultats
plt.plot(df["Année"], df["Vote_Candidat_A"], label="Historique", marker='o')
plt.scatter([2027], prediction, color='red', label="Prédiction 2027")
plt.xlabel("Année")
plt.ylabel("Vote Candidat A (%)")
plt.legend()
plt.show()
