"""Module de Pretraitement des Donnees

Fonctions:
- charger_donnees(): Charger les donnees depuis Kaggle
- nettoyer_donnees(): Nettoyer les valeurs impossibles
- separer_donnees(): Separer train/test
- creer_preprocesseur(): Creer le pipeline de pretraitement
- pipeline_pretraitement(): Pipeline complet
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import kagglehub
import os


# Constante pour la reproductibilité
RANDOM_STATE = 42

# ============================================================================
def charger_donnees(dataset_name="uciml/pima-indians-diabetes-database"):
    """
    Telecharge et charge le dataset depuis Kaggle.
    
    Parameters:
    -----------
    dataset_name : str
        Nom du dataset Kaggle (format: owner/dataset-name)
    
    Returns:
    --------
    pd.DataFrame
        Jeu de donnees charge
    """
    print("Telechargement du jeu de donnees depuis Kaggle...")
    
    path = kagglehub.dataset_download(dataset_name)
    
    csv_file = os.path.join(path, 'diabetes.csv')
    df = pd.read_csv(csv_file)
    
    print(f"Jeu de donnees charge: {df.shape[0]} lignes x {df.shape[1]} colonnes")
    print(f"Chemin: {csv_file}")
    
    return df

# ============================================================================
def nettoyer_donnees(df, verbose=True):
    """
    Nettoie les données en remplaçant les valeurs impossibles par NaN.
    
    Certaines variables ne peuvent pas physiologiquement être à 0:
    - Glucose, BloodPressure, SkinThickness, Insulin, BMI
    
    Parameters:
    -----------
    df : pd.DataFrame
        Jeu de donnees brut
    verbose : bool
        Afficher les informations de nettoyage
    
    Returns:
    --------
    pd.DataFrame
        Jeu de donnees nettoye
    """
    df_clean = df.copy()
    
    cols_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    if verbose:
        print("\nNettoyage des valeurs impossibles (0 -> NaN)")
        print("=" * 60)
    
    for col in cols_to_replace:
        if col in df_clean.columns:
            n_zeros = (df_clean[col] == 0).sum()
            df_clean[col] = df_clean[col].replace(0, np.nan)
            
            if verbose:
                pct = (n_zeros / len(df_clean)) * 100
                print(f"{col:20} : {n_zeros:3} zeros remplaces ({pct:.1f}%)")
    
    if verbose:
        print("=" * 60)
        print("Nettoyage termine\n")
    
    return df_clean

# ============================================================================
def separer_donnees(df, target_col='Outcome', test_size=0.2, random_state=RANDOM_STATE):
    """
    Sépare les données en ensembles d'entraînement et de test.
    
    Utilise un split stratifié pour préserver la distribution des classes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Jeu de donnees nettoye
    target_col : str
        Nom de la colonne cible
    test_size : float
        Proportion du jeu de test (0.0 à 1.0)
    random_state : int
        Seed pour la reproductibilité
    
    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"Separation des donnees:")
    print(f"   Train: {X_train.shape[0]} echantillons ({(1-test_size)*100:.0f}%)")
    print(f"   Test:  {X_test.shape[0]} echantillons ({test_size*100:.0f}%)")
    print(f"\n   Distribution dans le train:")
    print(f"   - Classe 0: {(y_train == 0).sum()} ({(y_train == 0).mean()*100:.1f}%)")
    print(f"   - Classe 1: {(y_train == 1).sum()} ({(y_train == 1).mean()*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

# ============================================================================
def creer_preprocesseur(strategy='median'):
    """
    Crée un pipeline de prétraitement sklearn.
    
    Le pipeline inclut:
    1. Imputation des valeurs manquantes
    2. Standardisation (scaling)
    
    Parameters:
    -----------
    strategy : str
        Stratégie d'imputation ('mean', 'median', 'most_frequent')
    
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Pipeline de pretraitement
    """
    preprocessor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=strategy)),
        ('scaler', StandardScaler())
    ])
    
    print(f"\nPipeline de pretraitement cree:")
    print(f"   1. Imputation des valeurs manquantes (strategie: {strategy})")
    print(f"   2. Standardisation (StandardScaler)")
    
    return preprocessor

# ============================================================================
def info_manquantes(df):
    """
    Affiche un résumé des valeurs manquantes.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Jeu de donnees à analyser
    
    Returns:
    --------
    pd.DataFrame
        Résumé des valeurs manquantes
    """
    missing_data = pd.DataFrame({
        'Colonne': df.columns,
        'Valeurs_Manquantes': df.isnull().sum(),
        'Pourcentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    
    # Trier par nombre de valeurs manquantes
    missing_data = missing_data.sort_values('Valeurs_Manquantes', ascending=False)
    missing_data = missing_data[missing_data['Valeurs_Manquantes'] > 0]
    
    return missing_data

# ============================================================================
def pipeline_pretraitement(dataset_name="uciml/pima-indians-diabetes-database", 
                       test_size=0.2, 
                       impute_strategy='median',
                       random_state=RANDOM_STATE):
    """
    Pipeline complet de prétraitement (fonction principale).
    
    Cette fonction encapsule toutes les étapes:
    1. Chargement des données
    2. Nettoyage
    3. Séparation train/test
    4. Création du preprocessor
    
    Parameters:
    -----------
    dataset_name : str
        Nom du dataset Kaggle
    test_size : float
        Proportion du jeu de test
    impute_strategy : str
        Stratégie d'imputation
    random_state : int
        Seed pour la reproductibilité
    
    Returns:
    --------
    dict
        Dictionnaire contenant:
        - 'X_train', 'X_test', 'y_train', 'y_test'
        - 'preprocessor'
        - 'df_raw', 'df_clean'
    """
    print("=" * 70)
    print("PIPELINE DE PRETRAITEMENT COMPLET")
    print("=" * 70)
    
    df_raw = charger_donnees(dataset_name)
    
    df_clean = nettoyer_donnees(df_raw, verbose=True)
    
    print("\nValeurs manquantes apres nettoyage:")
    print(info_manquantes(df_clean))
    
    X_train, X_test, y_train, y_test = separer_donnees(
        df_clean, 
        test_size=test_size, 
        random_state=random_state
    )
    
    preprocessor = creer_preprocesseur(strategy=impute_strategy)
    
    print("\n" + "=" * 70)
    print("PRETRAITEMENT TERMINE AVEC SUCCES")
    print("=" * 70)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'df_raw': df_raw,
        'df_clean': df_clean
    }


# ============================================================================
# Exemple d'utilisation
# ============================================================================

if __name__ == "__main__":
    data = pipeline_pretraitement()
    
    print("\n" + "=" * 70)
    print("RESUME DES DONNEES")
    print("=" * 70)
    print(f"X_train shape: {data['X_train'].shape}")
    print(f"X_test shape:  {data['X_test'].shape}")
    print(f"y_train shape: {data['y_train'].shape}")
    print(f"y_test shape:  {data['y_test'].shape}")
    
    print("\nDonnees pretes pour l'entrainement!")
    print("   Exemple:")
    print("   >>> from src.preprocess import pipeline_pretraitement")
    print("   >>> data = pipeline_pretraitement()")
    print("   >>> X_train, y_train = data['X_train'], data['y_train']")
