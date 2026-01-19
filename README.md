
#Prediction-of-Bottlenecks


## Présentation du projet

Ce projet a été développé dans le cadre de l'analyse du dataset "BPI Challenge 2017", portant sur un processus de demande de crédit bancaire. L'objectif est de transformer des données historiques (Event Logs) en un outil d'aide à la décision capable d'anticiper les goulots d'étranglement avant qu'ils ne surviennent.

Le système repose sur un réseau de neurones récurrents (LSTM) conçu pour effectuer deux tâches simultanées : prédire la prochaine activité d'un dossier et estimer le délai nécessaire pour y parvenir.

---

## Organisation du dépôt

Le projet est structuré de manière à séparer la phase de recherche (Notebooks), les actifs produits (Modèles) et l'outil de démonstration (Application).

### 1. Dossier Racine

* **app.py** : Le script principal de l'interface utilisateur. Il utilise le framework Streamlit pour offrir une démonstration interactive du modèle.
* **requirements.txt** : Liste des dépendances Python nécessaires (TensorFlow, Pandas, Scikit-learn, etc.) pour reproduire l'environnement d'exécution.
* **.gitignore** : Fichier de configuration excluant les données lourdes (fichiers .xes et .parquet) de ce dépôt pour respecter les limites de stockage de GitHub.

### 2. Dossier /notebooks

Ce dossier contient la démarche scientifique et technique étape par étape :

* **eda.IPYNB** : Analyse exploratoire des données. Ce notebook documente la compréhension du processus métier et l'identification statistique du seuil critique de 21 heures.
* **Preprocessing.ipynb** : Nettoyage et transformation des logs bruts. Il contient la logique de création des séquences temporelles utilisées pour entraîner l'intelligence artificielle.
* **Modeling.ipynb** : Architecture et entraînement du modèle LSTM. Il présente les courbes d'apprentissage et l'évaluation de la précision du modèle multi-tâches.

### 3. Dossier /models

Ce répertoire contient les fichiers générés lors de l'entraînement, indispensables au fonctionnement de l'application :

* **model_multi_task.keras** : Le modèle TensorFlow sauvegardé.
* **le_act.joblib** : L'encodeur d'activités permettant de traduire les prédictions numériques en noms d'étapes métier.
* **scaler_time.joblib** : Le normalisateur utilisé pour convertir les prédictions temporelles en heures réelles.
* **X_test.npy** : Un échantillon de données de test anonymisées permettant de faire fonctionner la démo sans charger l'intégralité du dataset.

---

## Données utilisées

Le fichier source original **BPI_Challenge_2017.xes** (565 Mo) n'est pas inclus dans ce dépôt par mesure d'optimisation. Il est toutefois possible de le récupérer via la source officielle :
[Lien vers le dataset officiel (4TU.ResearchData)](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884)

---

## Méthodologie et Interprétation

Le modèle considère qu'un goulot d'étranglement est détecté lorsque le délai prédit pour l'étape suivante dépasse **21 heures**. Ce seuil correspond au 90ème percentile des délais observés dans les données historiques.

L'application ne se contente pas de donner un chiffre ; elle interprète le flux. Par exemple, des répétitions d'étapes avec un délai de 0 heure sont analysées comme des activités de routine normales (appels téléphoniques successifs), tandis qu'un retour à une étape précédente avec un délai important est signalé comme une anomalie de "re-work" (retraitement).

---

