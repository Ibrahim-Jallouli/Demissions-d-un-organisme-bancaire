import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------- Chargement ----------
def charger_donnees(path1='data/table1.csv', path2='data/table2.csv'):
    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)
    return df1, df2

def afficher_anomalies(df, nom_table):
    print(f"\n===== ANALYSE DE {nom_table.upper()} =====")
    print(f"\n> Dimensions : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    print("\n> Valeurs manquantes par colonne :")
    print(df.isna().sum())

    if 'MTREV' in df.columns:
        print("\n> Revenus à zéro (MTREV) :")
        print((df['MTREV'] == 0).sum())

    if 'CDSEXE' in df.columns:
        print("\n> Valeurs uniques pour CDSEXE :")
        print(df['CDSEXE'].unique())

    if 'CDMOTDEM' in df.columns:
        print("\n> CDMOTDEM manquant ou vide :")
        vides = df['CDMOTDEM'].isna().sum() + (df['CDMOTDEM'].astype(str).str.strip() == '').sum()
        print(f"{vides} valeurs manquantes ou vides")

    if 'DTADH' in df.columns:
        print("\n> Exemples de dates (DTADH, DTDEM) :")
        print("DTADH:", df['DTADH'].head(3).tolist())
    if 'DTDEM' in df.columns:
        print("DTDEM:", df['DTDEM'].head(3).tolist())

    couples_redondants = [('AGEAD', 'RANGAGEAD'), ('AGEDEM', 'RANGAGEDEM'), ('ADH', 'RANGADH')]
    for exact, tranche in couples_redondants:
        if exact in df.columns and tranche in df.columns:
            print(f"\n> Présence du couple {exact} / {tranche} (potentielle redondance)")

    if 'CDSITFAM' in df.columns:
        print("\n> Valeurs uniques de la situation familiale (CDSITFAM) :")
        print(df['CDSITFAM'].unique())

    if 'CDTMT' in df.columns:
        print("\n> Valeurs uniques du statut sociétaire (CDTMT) :")
        print(df['CDTMT'].unique())

    if 'CDDEM' in df.columns:
        print("\n> Valeurs uniques du code de démission (CDDEM) :")
        print(df['CDDEM'].unique())

    for col in ['RANGAGEAD', 'RANGAGEDEM', 'RANGADH', 'RANGDEM']:
        if col in df.columns:
            print(f"\n> Valeurs uniques de la tranche {col} :")
            print(df[col].unique())

# ---------- Calculs temporels ----------
def calculer_adh(df):
    df['DTADH'] = pd.to_datetime(df['DTADH'], errors='coerce', dayfirst=True)
    df['DTDEM'] = pd.to_datetime(df['DTDEM'], errors='coerce', dayfirst=True)
    condition_actif = df['DTDEM'].dt.year == 1900
    df['ADH'] = None
    df.loc[condition_actif, 'ADH'] = 2007 - df.loc[condition_actif, 'DTADH'].dt.year
    df.loc[~condition_actif, 'ADH'] = df.loc[~condition_actif, 'DTDEM'].dt.year - df.loc[~condition_actif, 'DTADH'].dt.year
    df['ADH'] = pd.to_numeric(df['ADH'], downcast='integer').fillna(0)
    return df

def calculer_ages(df):
    if 'DTADH' in df.columns and 'DTNAIS' in df.columns:
        df['DTADH'] = pd.to_datetime(df['DTADH'], errors='coerce', dayfirst=True)
        df['DTNAIS'] = pd.to_datetime(df['DTNAIS'], errors='coerce', dayfirst=True)
        df['AGEAD'] = df['DTADH'].dt.year - df['DTNAIS'].dt.year

    if 'DTDEM' in df.columns and 'DTNAIS' in df.columns:
        df['DTDEM'] = pd.to_datetime(df['DTDEM'], errors='coerce', dayfirst=True)
        df['DTNAIS'] = pd.to_datetime(df['DTNAIS'], errors='coerce', dayfirst=True)
        df['AGEDEM'] = df['DTDEM'].dt.year - df['DTNAIS'].dt.year

    return df

# ---------- RANG helpers ----------
def nettoyer_rang_colonne(df, colonne):
    if colonne in df.columns:        
        df[colonne] = df[colonne].astype(str).str.extract(r'^(.{1,2})', expand=False)
    return df

def classer_rang_age_ad(age):
    try:
        age = float(age)
        return (
            '1' if 19 <= age <= 25 else
            '2' if 26 <= age <= 30 else
            '3' if 31 <= age <= 35 else
            '4' if 36 <= age <= 40 else
            '5' if 41 <= age <= 45 else
            '6' if 46 <= age <= 50 else
            '7' if 51 <= age <= 55 else
            '8' if age >= 56 else pd.NA
        )
    except:
        return pd.NA

def classer_rang_age_dem(age):
    try:
        age = float(age)
        return (
            '1' if 19 <= age <= 25 else
            '2' if 26 <= age <= 30 else
            '3' if 31 <= age <= 35 else
            '4' if 36 <= age <= 40 else
            '5' if 41 <= age <= 45 else
            '6' if 46 <= age <= 50 else
            '7' if 51 <= age <= 55 else
            '8' if 56 <= age <= 60 else
            '9' if 61 <= age <= 65 else
            'a' if 66 <= age <= 70 else
            'b' if age > 70 else pd.NA
        )
    except:
        return pd.NA

def classer_rang_adh(adh):
    try:
        adh = float(adh)
        return (
            '1' if 1 <= adh <= 4 else
            '2' if 5 <= adh <= 9 else
            '3' if 10 <= adh <= 14 else
            '4' if 15 <= adh <= 19 else
            '5' if 20 <= adh <= 24 else
            '6' if 25 <= adh <= 29 else
            '7' if 30 <= adh <= 34 else
            '8' if 35 <= adh <= 39 else
            '9' if adh >= 40 else pd.NA
        )
    except:
        return pd.NA

def completer_rang(df):
    if 'RANGAGEAD' in df.columns and 'AGEAD' in df.columns:
        df['RANGAGEAD'] = df.apply(
            lambda row: classer_rang_age_ad(row['AGEAD']) if pd.isna(row['RANGAGEAD']) and pd.notna(row['AGEAD']) else row['RANGAGEAD'],
            axis=1
        )
    if 'RANGAGEDEM' in df.columns and 'AGEDEM' in df.columns:
        df['RANGAGEDEM'] = df.apply(
            lambda row: classer_rang_age_dem(row['AGEDEM']) if pd.isna(row['RANGAGEDEM']) and pd.notna(row['AGEDEM']) else row['RANGAGEDEM'],
            axis=1
        )
    if 'RANGADH' in df.columns and 'ADH' in df.columns:
        df['RANGADH'] = df.apply(
            lambda row: classer_rang_adh(row['ADH']) if pd.isna(row['RANGADH']) and pd.notna(row['ADH']) else row['RANGADH'],
            axis=1
        )
    return df

# ---------- Revenus ----------
def corriger_revenus_par_famille(df):
    df['MTREV'] = pd.to_numeric(df['MTREV'], errors='coerce')
    revenu_par_famille = df[df['MTREV'] > 0].groupby('CDSITFAM')['MTREV'].mean()
    df['MTREV'] = df.apply(lambda row: revenu_par_famille.get(row['CDSITFAM'], 0) if row['MTREV'] == 0 else row['MTREV'], axis=1)
    return df, revenu_par_famille

def tracer_revenu_par_famille(revenu_par_famille, dossier='figures'):
    os.makedirs(dossier, exist_ok=True)
    plt.figure(figsize=(8, 5))

    revenu_par_famille = revenu_par_famille.sort_index()
    
    plt.plot(revenu_par_famille.index, revenu_par_famille.values, marker='o', linestyle='-')

    plt.title("Moyenne des revenus par situation familiale")
    plt.xlabel("Situation familiale (CDSITFAM)")
    plt.ylabel("Revenu moyen (€)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{dossier}/revenus_par_famille.png")
    plt.close()



# ---------- Fusion ----------
def fusionner_clients(df1, df2):
    df1['DEMISSIONNAIRE'] = 1
    df2['DTDEM'] = pd.to_datetime(df2['DTDEM'], errors='coerce', dayfirst=True)
    df2['DEMISSIONNAIRE'] = df2['DTDEM'].dt.year != 1900
    df2 = calculer_adh(df2)
    colonnes = list(set(df1.columns).union(set(df2.columns)))
    df1 = df1.reindex(columns=colonnes)
    df2 = df2.reindex(columns=colonnes)
    return pd.concat([df1, df2], ignore_index=True)

def supprimer_colonnes(df, colonnes_a_supprimer):
    colonnes_existantes = [col for col in colonnes_a_supprimer if col in df.columns]
    return df.drop(columns=colonnes_existantes)

# ---------- Main ----------
def main():
    df1, df2 = charger_donnees()
    afficher_anomalies(df1, "table1")
    afficher_anomalies(df2, "table2")
    df = fusionner_clients(df1, df2)
    df = calculer_ages(df)
    df = completer_rang(df)
    for col in ['RANGAGEAD', 'RANGAGEDEM', 'RANGADH']:
        df = nettoyer_rang_colonne(df, col)
    
    df, revenu_par_famille = corriger_revenus_par_famille(df)
    tracer_revenu_par_famille(revenu_par_famille)

    colonnes_ordonnees = [
        'ID','MTREV','CDSEXE', 'DTNAIS', 'NBENF', 'CDSITFAM',
        'DTADH', 'CDTMT', 'CDMOTDEM', 'CDCATCL', 'BPADH', 'DTDEM',
        'AGEAD', 'AGEDEM', 'ADH', 'RANGAGEAD', 'RANGAGEDEM', 'RANGADH',
        'RANGDEM', 'ANNEEDEM', 'CDDEM', 'DEMISSIONNAIRE'
    ]
    df = df[[col for col in colonnes_ordonnees if col in df.columns]]

    colonnes_a_supprimer = [
        'ID', 'BPADH', 'AGEAD', 'AGEDEM',
        'DTNAIS', 'DTADH', 'DTDEM', 'RANGDEM', 'ANNEEDEM', 'CDDEM', 'ADH'
    ]
    df = supprimer_colonnes(df, colonnes_a_supprimer)
    df.to_csv("data/clients_fusionnes.csv", index=False)

if __name__ == '__main__':
    main()
