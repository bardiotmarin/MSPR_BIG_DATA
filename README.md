# MSPR_BIG_DATA

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





Code du département,Libellé du département,Code du canton,Libellé du canton,Inscrits,Abstentions,% Abs/Ins,Votants,% Vot/Ins,Blancs,% Blancs/Ins,% Blancs/Vot,Nuls,% Nuls/Ins,% Nuls/Vot,Exprimés,% Exp/Ins,% Exp/Vot,N°Panneau,Sexe,Nom,Prénom,Voix,% Voix/Ins,% Voix/Exp,N°Panneau.1,Sexe.1,Nom.1,Prénom.1,Voix.1,% Voix/Ins.1,% Voix/Exp.1,N°Panneau.2,Sexe.2,Nom.2,Prénom.2,Voix.2,% Voix/Ins.2,% Voix/Exp.2,N°Panneau.3,Sexe.3,Nom.3,Prénom.3,Voix.3,% Voix/Ins.3,% Voix/Exp.3,N°Panneau.4,Sexe.4,Nom.4,Prénom.4,Voix.4,% Voix/Ins.4,% Voix/Exp.4,N°Panneau.5,Sexe.5,Nom.5,Prénom.5,Voix.5,% Voix/Ins.5,% Voix/Exp.5,N°Panneau.6,Sexe.6,Nom.6,Prénom.6,Voix.6,% Voix/Ins.6,% Voix/Exp.6,N°Panneau.7,Sexe.7,Nom.7,Prénom.7,Voix.7,% Voix/Ins.7,% Voix/Exp.7,N°Panneau.8,Sexe.8,Nom.8,Prénom.8,Voix.8,% Voix/Ins.8,% Voix/Exp.8,N°Panneau.9,Sexe.9,Nom.9,Prénom.9,Voix.9,% Voix/Ins.9,% Voix/Exp.9,N°Panneau.10,Sexe.10,Nom.10,Prénom.10,Voix.10,% Voix/Ins.10,% Voix/Exp.10,Département
