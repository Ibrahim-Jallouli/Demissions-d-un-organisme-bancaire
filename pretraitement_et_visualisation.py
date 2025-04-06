import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np

# ---------- Chargement ----------
def charger_donnees(path1='data/clients_fusionnes.csv'):
    df = pd.read_csv(path1)
    return df


def analyse_detaillee_finale(df):
    print("\n===== ANALYSE DÉTAILLÉE DE LA TABLE FINALE =====\n")
    
    total_rows = len(df)

    for col in df.columns:
        print(f"\n--- Attribut : {col} ---")
        print(f"Type : {df[col].dtype}")

        valeurs_nulles = df[col].isna().sum()
        valeurs_vides = (df[col] == '').sum() if df[col].dtype == 'object' else 0
        valeurs_aberrantes = (df[col].astype(str).str.lower().isin(['nan','none', 'null', '0000-00-00'])).sum()

        total_indefinies = valeurs_nulles + valeurs_vides + valeurs_aberrantes

        print(f"Valeurs uniques : {df[col].nunique(dropna=True)}")

        if pd.api.types.is_numeric_dtype(df[col]):
            print("\n> Statistiques descriptives :")
            print(df[col].describe())

            zeros = (df[col] == 0).sum()
            print(f"Valeurs à zéro : {zeros}")
        else:
            print("\n> Valeurs les plus fréquentes :")
            print(df[col].value_counts(dropna=False))

        print("-" * 50)


def traiter_rangagedem(df):
    df['RANGAGEDEM'] = df['RANGAGEDEM'].astype(str).str.strip().str.lower()
    df['RANGAGEDEM'] = df['RANGAGEDEM'].replace({'a': 10, 'b': 11})
    df['RANGAGEDEM'] = df['RANGAGEDEM'].replace("", 0)
    df['RANGAGEDEM'] = pd.to_numeric(df['RANGAGEDEM'], errors='coerce')
    df['RANGAGEDEM'].fillna(0, inplace=True)
    print("Traitement de RANGAGEDEM terminé.")
    return df

def imputer_mode(df, col):
    df[col] = df[col].astype(str).str.strip().str.lower()
    valeurs_manquantes = ['na', 'nan', 'none', '', 'null', '<n', 'n/a', '0000-00-00']
    df[col] = df[col].replace(valeurs_manquantes, np.nan)
    df[col] = pd.to_numeric(df[col], errors='coerce')
    mode_value = df[col].mode().iloc[0]
    df[col] = df[col].fillna(mode_value)

    return df


def encoder_ordinal(df, col):
    encoder = OrdinalEncoder()
    df[[col]] = encoder.fit_transform(df[[col]].astype(str))
    return df, encoder

def encoder_cdmotdem(df):
    categories = ['DV', 'DA', 'RA', 'DC']
    df['CDMOTDEM'] = df['CDMOTDEM'].where(df['CDMOTDEM'].isin(categories), np.nan)
    encoder = OneHotEncoder(sparse_output=False, categories=[categories], handle_unknown='ignore')
    encoded_array = encoder.fit_transform(df[['CDMOTDEM']])

    encoded_cols = [f'CDMOTDEM_{cat}' for cat in categories]
    df_encoded = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)

    # Fusionner
    df = pd.concat([df.drop(columns=['CDMOTDEM']), df_encoded], axis=1)

    print("CDMOTDEM encodé (DV, DA, RA, DC ).")
    return df


def normaliser_revenu(df):
    scaler = StandardScaler()
    df[['MTREV']] = scaler.fit_transform(df[['MTREV']]).round(4)
    print("MTREV normalisé (centré-réduit).")
    return df, scaler

def generer_matrice_correlation(df, output_dir='figures'):
    os.makedirs(output_dir, exist_ok=True)

    numeric_cols = df.select_dtypes(include='number')

    corr_matrix = numeric_cols.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Matrice de corrélation")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/matrice_correlation.png')
    plt.close()
    
    print(f"Matrice de corrélation sauvegardée dans : {output_dir}")

def generer_histogrammes(df, output_dir='figures/histogrammes-attributs'):
    os.makedirs(output_dir, exist_ok=True)
    colonnes_numeriques = df.select_dtypes(include='number').columns

    for col in colonnes_numeriques:
        plt.figure(figsize=(6, 4))


        if col == 'MTREV':
            seuil = df[col].quantile(0.99)
            data = df[df[col] <= seuil][col]
        else:
            data = df[col]

        sns.histplot(data, bins=30, kde=False)
        plt.title(f"Histogramme de {col}")
        plt.xlabel(col)
        plt.ylabel("Fréquence")
        plt.ticklabel_format(style='plain', axis='x')  # meilleure lisibilité
        plt.tight_layout()
        plt.savefig(f"{output_dir}/hist_{col}.png")
        plt.close()

def generer_boxplots(df, output_dir='figures/boxplots-attributs'):
    os.makedirs(output_dir, exist_ok=True)
    colonnes_numeriques = df.select_dtypes(include='number').columns
    colonnes_numeriques = [col for col in colonnes_numeriques if col != 'DEMISSIONNAIRE']

    for col in colonnes_numeriques:
        plt.figure(figsize=(6, 4))

        if col == 'MTREV':
            seuil = df[col].quantile(0.99)
            data = df[df[col] <= seuil]
            print(f"[INFO] Boxplot de {col} limité à {seuil:.2f} € (99e percentile)")
        else:
            data = df

        sns.boxplot(y=data[col])
        plt.title(f"Boxplot de {col}")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/box_{col}.png")
        plt.close()


def main():
    df = charger_donnees()
    traiter_rangagedem(df)
    imputer_mode(df, 'RANGAGEAD')
    imputer_mode(df, 'RANGADH')

    df, enc = encoder_ordinal(df, 'CDCATCL')
    analyse_detaillee_finale(df)
    mapping = {
    'A': 1, 'B': 2, 'M': 3, 'C': 4, 'D': 5,
    'U': 6, 'S': 7, 'V': 8, 'E': 9, 'G': 10,
    'P': 11, 'F': 12
    }
    df['CDSITFAM'] = df['CDSITFAM'].map(mapping)
    generer_histogrammes(df)
    generer_boxplots(df)
    df = encoder_cdmotdem(df)
    print(df[df['MTREV'] > 100000][['MTREV', 'CDSITFAM', 'CDTMT']])

  
    df, _ = normaliser_revenu(df)



    ordre_colonnes = [
        'MTREV','CDSEXE', 'NBENF', 'CDSITFAM', 'CDCATCL','CDTMT', 'RANGAGEAD',
        'RANGAGEDEM', 'RANGADH', 'CDMOTDEM_DA', 'CDMOTDEM_DC',
        'CDMOTDEM_DV', 'CDMOTDEM_RA', 'DEMISSIONNAIRE'
    ]

    df = df[[col for col in ordre_colonnes if col in df.columns]]

    generer_matrice_correlation(df)





    df.to_csv("data/clients_model_ready.csv", index=False)



if __name__ == '__main__':
    main()
