# MSPR_BIG_DATA
<video width="320" height="240" controls>
  <source src="src\img\banner.mp4" type="video/mp4">
  Votre navigateur ne supporte pas la vidéo.
</video>

install requirements : 
pip install -r requirements.txt
start minio


Bdd login term with psql : psql postgresql://user:password@localhost:5433/mspr_warehouse





# MSPR Big Data & Analyse de Données

## Description du Projet

Ce projet a été réalisé dans le cadre de la MSPR Big Data et Analyse de Données. 
L'objectif principal est de concevoir une preuve de concept (POC) permettant de prédire les tendances électorales grâce à l'intelligence artificielle. 
Le projet s'appuie sur des données socio-économiques et électorales pour établir des corrélations, créer un modèle prédictif supervisé et fournir des visualisations exploitables.

---

## Fonctionnalités Principales

1. **ETL (Extract, Transform, Load)** :
   - Extraction des données brutes à partir de fichiers Excel et CSV.
   - Transformation des données pour les normaliser et les rendre exploitables.
   - Chargement des données dans un datalake MinIO et une base de données SQL.

2. **Modélisation Prédictive** :
   - Création d'un modèle supervisé à partir des jeux d'entraînement et de test.
   - Prédictions sur les tendances électorales à 1, 2 et 3 ans.

3. **Visualisation des Données** :
   - Graphiques interactifs pour illustrer les corrélations et les prédictions.
   - Tableau de bord PowerBI ou visualisations Python (Matplotlib/Seaborn).

---

## Structure du Projet

MSPR_BIG_DATA/
│
├── data/
│ ├── raw/ # Données brutes (Excel, CSV)
│ └── processed/ # Données transformées
├── ML/
│ ├── models/ # Modèles prédictifs
│
├── src/
│ ├── etl.py # Script ETL principal
│ ├── .env # Variables d'environnement (MinIO, SQL)
│ ├── utils.py # Fonctions utilitaires (connexion MinIO, SQLAlchemy)
│ └── models/ # Modèles prédictifs
│
├── main.py # Point d'entrée du projet
├── .env # Variables d'environnement (MinIO, SQL)
├── requirements.txt # Dépendances Python
└── README.md # Documentation du projet



---

## Installation

### Prérequis

- **Python 3.9+**
- **Docker & Docker Compose**
- **PostgreSQL**
- **MinIO**

### Étapes d'installation

1. Clonez le dépôt :

```git clone https://github.com/bardiotmarin/MSPR_BIG_DATA.git ```

cd mspr_big_data 



2. Installez les dépendances Python :
``` pip install -r requirements.txt ```



3. Configurez les variables d'environnement dans le fichier `.env` :
MINIO_ENDPOINT=http://127.0.0.1:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
BUCKET_NAME=datalake

DB_HOST=localhost
DB_PORT=5432
DB_NAME=election_warehouse
DB_USER=postgres
DB_PASSWORD=password


4. Lancez MinIO et PostgreSQL avec Docker Compose :
```docker-compose up -d ```


5. Initialisez le projet et exécutez le pipeline ETL :
```python main.py```


---

## Utilisation

1. Placez vos fichiers bruts dans le dossier `data/raw/`.
2. Exécutez `main.py` pour lancer le pipeline ETL.
3. Accédez aux données transformées dans MinIO ou PostgreSQL.
4. Visualisez les résultats via PowerBI ou les scripts Python fournis.

---

## Livrables

- **Données traitées** : Disponibles dans MinIO (`datalake`) et PostgreSQL (`data Warehouse`) .
- **Modèle prédictif** : Précision mesurée avec métriques comme l'accuracy.
- **Visualisations** : Graphiques interactifs pour l'analyse décisionnelle.

---

## Exemple de Résultats Attendus

1. Corrélations entre indicateurs socio-économiques et résultats électoraux.
2. Prédictions sur les tendances électorales à 1, 2 et 3 ans.
3. Tableau de bord interactif pour explorer les données.

---

## Auteur

Projet réalisé par Marin Bardiot , Léo Drouill ,  dans le cadre de la MSPR TPRE813.

---

## Ressources Supplémentaires

- [Documentation officielle MinIO](https://docs.min.io)
- [Documentation officielle PostgreSQL](https://www.postgresql.org/docs/)
- [Données publiques](https://www.data.gouv.fr)

---

## Bonus : Musique pour la concentration 🎵

<iframe title="deezer-widget" src="https://widget.deezer.com/widget/auto/track/3059826821" width="100%" height="300" frameborder="0" allowtransparency="true" allow="encrypted-media; clipboard-write"></iframe>

