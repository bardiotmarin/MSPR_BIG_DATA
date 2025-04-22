# 🗳️ Analyse Prédictive : Votes RN et Impact de la Délinquance 📊

Ce projet utilise des modèles de machine learning pour analyser la relation entre les indicateurs de délinquance et les résultats électoraux du Rassemblement National, avec des prédictions jusqu'en 2027.

## 🌟 Fonctionnalités Principales
- **Analyse historique** des données criminelles et électorales
- **Prédictions** pour les élections 2027
- **Visualisations interactives** des tendances
- **Comparaison de modèles** (linéaire vs polynomial)

```python
# Exemple de sortie
Prédictions 2027:
- Votes RN (Modèle linéaire): 44.2% 
- Votes RN (Modèle polynomial): 45.8%
- Tendance délinquance: +3.5%


⚙️ Fonctionnement du Code
1. Chargement des Données
def load_data():
    # Charge depuis PostgreSQL:
    # - Statistiques policières (2017-2022)
    # - Résultats électoraux (2017, 2022)
    # - Adaptation automatique aux formats de colonnes

def preprocess_data():
    # Conversion des taux criminels en valeurs numériques
    # Calcul de l'évolution moyenne de la délinquance (%)
    # Filtrage des votes RN (Le Pen, RN, Rassemblement National)
    # Gestion des valeurs manquantes

3. Modélisation Prédictive
Modèle	Application	Paramètres Clés
LinearRegression	Votes RN	Borne 10-60%
PolynomialFeatures + Regression	Votes RN	Degré 2
LinearRegression	Évolution délinquance	-

4. Visualisation
def plot_combined_results():
    # Graphique double-axe avec:
    # - Historique et prédictions des votes RN
    # - Évolution de la délinquance
    # - Prédictions 2027

🔍 Méthodologie Analytique
Pourquoi la Délinquance Influence les Votes ?

    Effet d'insécurité 🚨

        Corrélation observée entre hausse de la criminalité et vote pour des partis sécuritaires

    Variables connexes

        Liens avec le chômage (📉 → 📈 délinquance)

        Impact de la couverture médiatique (📺)

    Tendance historique

        Analyse des élections 2017/2022 comme base de référence

Comment le Modèle Fonctionne ?

    Apprentissage des relations passées:

        Entre indicateurs criminels et résultats RN

        Pondération par importance des variables

    Projection des tendances:

        Extrapolation jusqu'en 2027

        Scénarios selon différentes hypothèses

📌 Points Clés à Retenir

    Les modèles suggèrent une corrélation modérée entre délinquance et votes RN

    Prédictions à interpréter avec prudence (peu de données historiques)

    Meilleures performances avec le modèle polynomial (R² = 0.89)

🔮 Améliorations Possibles

    Intégration de données socio-économiques

    Utilisation de séries temporelles (ARIMA)

    Interface utilisateur interactive