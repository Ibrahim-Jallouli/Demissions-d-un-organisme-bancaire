# Prédiction de Démission des Clients Dépendances

## Structure du projet


├── data/								# Contient les fichiers CSV 
│   ├── table1.csv
│   ├── table2.csv
│   ├── clients_fusionnes.csv
│   └── clients_model_ready.csv
├── figures/							# Graphiques d'analyse (histogrammes, boxplots, etc.)		
├── models/								# Modèles entraînés, matrices de confusion, fichiers CSV de résultats
├── fusion_nettoyage_clients.py
├── pretraitement_et_visualisation.py
├── modele_prediction.py
├── main.py								# Pipeline principal qui exécute tout


# Dépendances

pandas
numpy
matplotlib
seaborn
scikit-learn
imblearn
joblib

# Installation et Exécution :

	# Extraire le fichier ZIP
	unzip BI.zip 
	cd BI

	# Créer un environnement virtuel
	python3 -m venv venv
	source venv/bin/activate

	# Installer les dépendances nécessaires
	pip install pandas numpy matplotlib seaborn scikit-learn imblearn joblib

	# Lancer le script principal
	python3 main.py
