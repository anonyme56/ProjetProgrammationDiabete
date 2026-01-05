"""Module d'Entrainement des Modeles

Fonctions:
- entrainer_modeles_baseline(): Entrainer plusieurs modeles baseline
- optimiser_random_forest(): Optimiser Random Forest
- optimiser_gradient_boosting(): Optimiser Gradient Boosting
- sauvegarder_modele(): Sauvegarder un modele
- charger_modele(): Charger un modele
- entrainer_et_optimiser_tout(): Pipeline complet
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report
)

import joblib
import os

RANDOM_STATE = 42

# ============================================================================
def entrainer_modeles_baseline(X_train, y_train, preprocessor, random_state=RANDOM_STATE):
    """
    Entraine plusieurs modeles de classification (baseline).
    
    Modeles entraines:
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - SVM (RBF kernel)
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Features d'entrainement
    y_train : pd.Series
        Target d'entrainement
    preprocessor : sklearn.pipeline.Pipeline
        Pipeline de pretraitement
    random_state : int
        Seed pour la reproductibilite
    
    Returns:
    --------
    dict
        Dictionnaire {nom_modele: pipeline_entraine}
    """
    print("=" * 70)
    print("ENTRAINEMENT DES MODELES BASELINE")
    print("=" * 70)
    
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=random_state, 
            max_iter=1000
        ),
        'Random Forest': RandomForestClassifier(
            random_state=random_state, 
            n_estimators=100
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            random_state=random_state, 
            n_estimators=100
        ),
        'SVM': SVC(
            random_state=random_state, 
            probability=True, 
            kernel='rbf'
        )
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nEntrainement: {name}")
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=5, scoring='roc_auc'
        )
        
        print(f"   Entraine avec succes")
        print(f"   Validation Croisee (AUC): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        trained_models[name] = pipeline
    
    print("\n" + "=" * 70)
    print(f"{len(trained_models)} MODELES ENTRAINES AVEC SUCCES")
    print("=" * 70)
    
    return trained_models

# ============================================================================
def optimiser_random_forest(X_train, y_train, preprocessor, 
                          n_jobs=-1, random_state=RANDOM_STATE):
    """
    Parameters:
    -----------
    X_train : pd.DataFrame
        Variables d'entrainement
    y_train : pd.Series
        Cible d'entrainement
    preprocessor : sklearn.pipeline.Pipeline
        Pipeline de pretraitement
    n_jobs : int
        Nombre de CPU à utiliser (-1 = tous)
    random_state : int
        Seed pour la reproductibilite
    
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Meilleur modele optimise
    """
    print("\n" + "=" * 70)
    print("OPTIMISATION - RANDOM FOREST (GridSearchCV)")
    print("=" * 70)
    
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=random_state))
    ])
    
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4],
        'classifier__max_features': ['sqrt', 'log2']
    }
    
    n_combinations = np.prod([len(v) for v in param_grid.values()])
    print(f"\nEspace de recherche: {n_combinations} combinaisons")
    print(f"   Validation croisee: 5-fold")
    print(f"   Metrique: AUC-ROC")
    
    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=n_jobs,
        verbose=1
    )
    
    print(f"\nRecherche en cours (cela peut prendre quelques minutes)...")
    grid_search.fit(X_train, y_train)
    
    print(f"\nOptimisation terminee!")
    print(f"\nMeilleurs hyperparametres:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    print(f"\nMeilleur score CV (AUC): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# ============================================================================
def optimiser_gradient_boosting(X_train, y_train, preprocessor, 
                               n_iter=50, n_jobs=-1, random_state=RANDOM_STATE):
    """
    Parameters:
    -----------
    X_train : pd.DataFrame
        Variables d'entrainement
    y_train : pd.Series
        Cible d'entrainement
    preprocessor : sklearn.pipeline.Pipeline
        Pipeline de pretraitement
    n_iter : int
        Nombre d'iterations de la recherche aleatoire
    n_jobs : int
        Nombre de CPU à utiliser
    random_state : int
        Seed pour la reproductibilite
    
    Returns:
    --------
    sklearn.pipeline.Pipeline
        Meilleur modele optimise
    """
    print("\n" + "=" * 70)
    print("OPTIMISATION - GRADIENT BOOSTING (RandomizedSearchCV)")
    print("=" * 70)
    
    gb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', GradientBoostingClassifier(random_state=random_state))
    ])
    
    param_distributions = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2],
        'classifier__max_depth': [3, 5, 7],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__subsample': [0.8, 0.9, 1.0]
    }
    
    n_total = np.prod([len(v) for v in param_distributions.values()])
    print(f"\nEspace de recherche: {n_total} combinaisons possibles")
    print(f"   Recherche aleatoire: {n_iter} iterations")
    print(f"   Validation croisee: 5-fold")
    print(f"   Metrique: AUC-ROC")
    
    random_search = RandomizedSearchCV(
        estimator=gb_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='roc_auc',
        n_jobs=n_jobs,
        verbose=1,
        random_state=random_state
    )
    
    print(f"\nRecherche en cours...")
    random_search.fit(X_train, y_train)
    
    print(f"\nOptimisation terminee!")
    print(f"\nMeilleurs hyperparametres:")
    for param, value in random_search.best_params_.items():
        print(f"   {param}: {value}")
    
    print(f"\nMeilleur score CV (AUC): {random_search.best_score_:.4f}")
    
    return random_search.best_estimator_

# ============================================================================
def sauvegarder_modele(model, filename, directory='models'):
    """
    Parameters:
    -----------
    model : sklearn model
        Modele à sauvegarder
    filename : str
        Nom du fichier (sans extension)
    directory : str
        Dossier de destination
    
    Returns:
    --------
    str
        Chemin complet du fichier sauvegarde
    """
    os.makedirs(directory, exist_ok=True)
    
    if not filename.endswith('.pkl'):
        filename += '.pkl'
    
    filepath = os.path.join(directory, filename)
    
    joblib.dump(model, filepath)
    
    print(f"\nModele sauvegarde: {filepath}")
    
    return filepath

# ============================================================================
def charger_modele(filepath):
    """
    Charge un modele sauvegarde.
    
    Parameters:
    -----------
    filepath : str
        Chemin vers le fichier .pkl
    
    Returns:
    --------
    sklearn model
        Modele charge
    """
    model = joblib.load(filepath)
    print(f"Modele charge depuis: {filepath}")
    return model

# ============================================================================
def entrainer_et_optimiser_tout(X_train, y_train, preprocessor, 
                           optimize=True, save=True, random_state=RANDOM_STATE):
    """
    Parameters:
    -----------
    X_train : pd.DataFrame
        Variables d'entrainement
    y_train : pd.Series
        Cible d'entrainement
    preprocessor : sklearn.pipeline.Pipeline
        Pipeline de pretraitement
    optimize : bool
        Si True, optimise RF et GB
    save : bool
        Si True, sauvegarde les meilleurs modeles
    random_state : int
        Seed pour la reproductibilite
    
    Returns:
    --------
    dict
        Dictionnaire contenant tous les modeles
    """
    print("\n" + "=" * 70)
    print("PIPELINE COMPLET D'ENTRAINEMENT")
    print("=" * 70)
    
    results = {}
    
    baseline_models = entrainer_modeles_baseline(
        X_train, y_train, preprocessor, random_state
    )
    results['baseline'] = baseline_models
    
    if optimize:
        best_rf = optimiser_random_forest(
            X_train, y_train, preprocessor, random_state=random_state
        )
        results['Random Forest (Tuned)'] = best_rf
        
        best_gb = optimiser_gradient_boosting(
            X_train, y_train, preprocessor, random_state=random_state
        )
        results['Gradient Boosting (Tuned)'] = best_gb
    
    if save and optimize:
        sauvegarder_modele(best_rf, 'random_forest_tuned.pkl')
        sauvegarder_modele(best_gb, 'gradient_boosting_tuned.pkl')
        sauvegarder_modele(best_gb, 'best_diabetes_model.pkl')
    
    print("\n" + "=" * 70)
    print("ENTRAINEMENT COMPLET TERMINE")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    from preprocess import pipeline_pretraitement
    
    data = pipeline_pretraitement()
    
    models = entrainer_et_optimiser_tout(
        X_train=data['X_train'],
        y_train=data['y_train'],
        preprocessor=data['preprocessor'],
        optimize=True,
        save=True
    )
    
    print("\nModeles disponibles:")
    for name in models.keys():
        print(f"   - {name}")
