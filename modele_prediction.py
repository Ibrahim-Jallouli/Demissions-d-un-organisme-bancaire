import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import seaborn as sns
import joblib
from imblearn.over_sampling import RandomOverSampler
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import time

# ---------- Chargement des données ----------
def charger_donnees(path='data/clients_model_ready.csv'):
    return pd.read_csv(path)

# ---------- Affichage de la matrice de confusion ----------
def afficher_matrice_confusion(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Matrice de confusion - {model_name}")
    os.makedirs('figure', exist_ok=True)
    plt.savefig(f'models/figures/confusion_matrix_{model_name.lower()}.png')
    plt.close()

# ---------- Sauvegarder le modèle, CSV et graphes ----------
def sauvegarder_modele(model, nom_modele, y_proba, y_pred_proba, y_valid, nom_fichier_csv):
    joblib.dump(model, f'models/{nom_modele}.pkl')
    print(f"Modèle {nom_modele} sauvegardé avec succès !")
    result_df = pd.DataFrame({
        'Probabilité_démission': y_proba,
        'Classe_prévue': y_pred_proba,
        'Classe_réelle': y_valid.reset_index(drop=True)
    })
    result_df.to_csv(f"models/CSV/{nom_fichier_csv}", index=False)

    plt.figure(figsize=(8, 6))
    plt.hist(y_proba, bins=20, color='darkorange', edgecolor='black')  
    plt.xlabel("Probabilité de démission")
    plt.ylabel("Nombre de clients")
    plt.title(f"Distribution des probabilités de démission ({nom_modele} - validation)")
    plt.tight_layout()
    plt.savefig(f"models/figures/hist_proba_{nom_modele}_validation.png")
    plt.close()

# ---------- Fonctions d'entraînement des modèles ----------
def entrainer_model(X_train, y_train, X_valid, y_valid, model, nom_modele):

    print(f"Entraînement du modèle {nom_modele} en cours...")
    start = time.time()
    # Entraînement du modèle
    model.fit(X_train, y_train)

    end = time.time()
    print(f"{nom_modele} entraîné en {end - start:.2f} secondes.\n")

    # Prédictions et probabilités
    y_proba = model.predict_proba(X_valid)[:, 1]
    y_pred_proba = (y_proba >= 0.6).astype(int)

    print(f"=== Évaluation {nom_modele} (validation) ===")
    print("Accuracy :", accuracy_score(y_valid, y_pred_proba))
    print("F1 Score :", f1_score(y_valid, y_pred_proba))
    print("Recall   :", recall_score(y_valid, y_pred_proba))

    afficher_matrice_confusion(y_valid, y_pred_proba, nom_modele)
    plot_decision_validation_3D(X_valid, y_valid, y_pred_proba, nom_modele)


    return model

# ---------- Évaluation finale sur jeu de test ----------
def evaluer_modele_final(model, X_test, y_test, nom_modele):
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    y_pred = (y_proba >= 0.6).astype(int) if y_proba is not None else model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Matrice de confusion - {nom_modele} (test)")
    plt.savefig(f'models/figures/confusion_matrix_test_{nom_modele.lower()}.png')
    plt.close()

    if y_proba is not None:
        result_df = pd.DataFrame({
            'Probabilité_démission': y_proba,
            'Classe_prévue': y_pred,
            'Classe_réelle': y_test.reset_index(drop=True)
        })
        os.makedirs("models", exist_ok=True)
        result_df.to_csv(f"models/CSV/probabilites_{nom_modele.lower()}_test.csv", index=False)

        plt.figure(figsize=(8, 6))
        plt.hist(y_proba, bins=20, color='lightcoral', edgecolor='black')
        plt.xlabel("Probabilité de démission")
        plt.ylabel("Nombre de clients")
        plt.title(f"Distribution des probabilités de démission ({nom_modele} - test)")
        plt.tight_layout()
        plt.savefig(f"models/figures/hist_proba_{nom_modele.lower()}_test.png")
        plt.close()
    

def plot_decision_validation_3D(X_valid, y_valid, y_pred, model_name):

    print (f"=== Visualisation 3D {model_name} (validation) ===")
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_valid)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Vert = bien classé, Rouge = erreur
    colors = np.where(y_pred == y_valid, 'green', 'red')
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
               c=colors, alpha=0.6, edgecolor='k')


    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    ax.set_zlabel('PCA 3')
    ax.set_title(f'Classification validation 3D - {model_name}')

    # Légende
    legend_elements = [
        mpatches.Patch(color='green', label='Bonne prédiction'),
        mpatches.Patch(color='red', label='Mauvaise prédiction')
    ]
    ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(f'models/figures/pca3d_val_{model_name}.png')
    plt.close()



def plot_importance(importance, model_name):
    # Trier par valeur absolue du coefficient
    print("=== Importance des attributs en cours ===")
    importance = importance.reindex(importance['Coefficient'].abs().sort_values().index)
    plt.figure(figsize=(10, 8))
    plt.barh(importance['Feature'], importance['Coefficient'], color='skyblue')
    plt.xlabel('Coefficient')
    plt.title(f'Importance des attributs - {model_name}')
    plt.savefig(f'{model_name}_feature_importance.png')
    plt.close()

def importance_logistic_regression(model, feature_names):
    importance = pd.DataFrame({'Feature': feature_names, 'Coefficient': model.coef_[0]})
    plot_importance(importance, 'models/figures/Logistic_Regression')

def importance_knn(model, X_valid, y_valid, feature_names):
    result = permutation_importance(model, X_valid, y_valid, n_repeats=10, random_state=42, scoring='accuracy')
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': result.importances_mean
    })
    plot_importance(importance, 'models/figures/KNN')


def get_feature_importance(model, X_valid, y_valid, feature_names, model_name):
    result = permutation_importance(model, X_valid, y_valid, n_repeats=10, random_state=42, scoring='accuracy')
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean
    }).sort_values(by='Importance', ascending=True)

    # Affichage
    plt.figure(figsize=(10, 6))
    plt.barh(importance['Feature'], importance['Importance'], color='mediumseagreen')
    plt.xlabel("Importance (chute de performance)")
    plt.title(f"Importance des attributs - {model_name}")
    plt.tight_layout()
    plt.savefig(f'models/figures/{model_name}_permutation_importance.png')
    plt.show()


# ---------- Main ---------- 
def main():
    df = charger_donnees()
    print("Répartition de la variable cible :")
    print(df['DEMISSIONNAIRE'].value_counts(normalize=True) * 100)

    # Préparation des données
    X = df.drop(columns=['DEMISSIONNAIRE', 'CDMOTDEM_DA','CDMOTDEM_DC','CDMOTDEM_DV','CDMOTDEM_RA','RANGAGEDEM'])
    y = df['DEMISSIONNAIRE']

    # Découpage des données
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    ros = RandomOverSampler(random_state=42)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    print("Répartition de la variable cible après suréchantillonnage :")
    print(y_train_resampled.value_counts(normalize=True) * 100)
    # Entraînement et évaluation des modèles
    model_lr = entrainer_model(X_train, y_train, X_valid, y_valid, LogisticRegression(solver='liblinear', C=0.1)
, 'LogisticRegression')
    model_svm = entrainer_model(X_train, y_train, X_valid, y_valid, SVC(kernel='rbf', probability=True), 'SVM')
    model_nb = entrainer_model(X_train, y_train, X_valid, y_valid, CategoricalNB(), 'NaiveBayes')
    model_knn = entrainer_model(X_train, y_train, X_valid, y_valid, KNeighborsClassifier(n_neighbors=10), 'KNN')
   

    # Évaluation finale sur le jeu de test
    evaluer_modele_final(model_lr, X_test, y_test, "LogisticRegression")
    evaluer_modele_final(model_nb, X_test, y_test, "NaiveBayes")
    evaluer_modele_final(model_knn, X_test, y_test, "KNN")
    evaluer_modele_final(model_svm, X_test, y_test, "SVM")
       

    # Importance des attributs
    importance_logistic_regression(model_lr, X.columns)
    importance_knn(model_knn, X_valid, y_valid, X.columns)
    # get_feature_importance(model_svm, X_valid, y_valid, X.columns, 'SVM_RBF')
    # get_feature_importance(model_nb, X_valid, y_valid, X.columns, 'NaiveBayes')

if __name__ == '__main__':
    main()
