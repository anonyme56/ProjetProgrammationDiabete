"""Module d'Evaluation des Modeles

Fonctions:
- evaluer_modele(): Evaluer un modele sur le test set
- comparer_modeles(): Comparer plusieurs modeles
- afficher_matrice_confusion(): Matrice de confusion
- afficher_courbes_roc(): Courbes ROC
- afficher_importance_features(): Importance des variables
- generer_rapport(): Rapport complet d'evaluation
- predire_nouvel_echantillon(): Prediction sur nouvel echantillon
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)

import os

# ============================================================================
def evaluer_modele(model, X_test, y_test, model_name="Model"):
    """
    Parameters:
    -----------
    model : sklearn model
        Modele entraine
    X_test : pd.DataFrame
        Variables de test
    y_test : pd.Series
        Cible de test
    model_name : str
        Nom du modele pour l'affichage
    
    Returns:
    --------
    dict
        Dictionnaire contenant toutes les metriques
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    print("=" * 70)
    print(f"EVALUATION - {model_name}")
    print("=" * 70)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1_score']:.4f}")
    print(f"AUC-ROC:   {metrics['roc_auc']:.4f}")
    print("\n" + classification_report(
        y_test, y_pred, 
        target_names=['Pas de diabete', 'Diabete']
    ))
    
    return metrics

# ============================================================================
def comparer_modeles(models_dict, X_test, y_test):
    """
    Parameters:
    -----------
    models_dict : dict
        Dictionnaire {nom_modele: modele_entraine}
    X_test : pd.DataFrame
        Variables de test
    y_test : pd.Series
        Cible de test
    
    Returns:
    --------
    pd.DataFrame
        Tableau comparatif des performances
    """
    print("\n" + "=" * 70)
    print("COMPARAISON DES MODELES")
    print("=" * 70)
    
    results = []
    
    for name, model in models_dict.items():
        print(f"\nEvaluation de: {name}")
        metrics = evaluer_modele(model, X_test, y_test, name)
        
        results.append({
            'Modele': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score'],
            'AUC-ROC': metrics['roc_auc']
        })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('AUC-ROC', ascending=False)
    
    print("\n" + "=" * 70)
    print("TABLEAU RECAPITULATIF")
    print("=" * 70)
    print(df_results.to_string(index=False))
    
    best_model = df_results.iloc[0]
    print(f"\nMeilleur modele: {best_model['Modele']}")
    print(f"   AUC-ROC: {best_model['AUC-ROC']:.4f}")
    
    return df_results

# ============================================================================
def afficher_matrice_confusion(y_test, y_pred, model_name="Model", save=False, save_dir='reports/figures'):
    """
    Affiche la matrice de confusion.
    
    Parameters:
    -----------
    y_test : array-like
        Vraies valeurs
    y_pred : array-like
        Predictions
    model_name : str
        Nom du modele
    save : bool
        Sauvegarder la figure
    save_dir : str
        Dossier de sauvegarde
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pas de diabete', 'Diabete'],
                yticklabels=['Pas de diabete', 'Diabete'])
    plt.title(f'Matrice de Confusion - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Predite')
    plt.tight_layout()
    
    if save:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardee: {filepath}")
    
    plt.show()

# ============================================================================
def afficher_courbes_roc(models_dict, X_test, y_test, save=False, save_dir='reports/figures'):
    """
    Parameters:
    -----------
    models_dict : dict
        Dictionnaire {nom_modele: modele}
    X_test : pd.DataFrame
        Variables de test
    y_test : pd.Series
        Cible de test
    save : bool
        Sauvegarder la figure
    save_dir : str
        Dossier de sauvegarde
    """
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Hasard (AUC = 0.500)')
    
    plt.xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
    plt.ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
    plt.title('Courbes ROC - Comparaison des Modeles', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'roc_curves_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardee: {filepath}")
    
    plt.show()

# ============================================================================
def afficher_importance_features(model, feature_names, model_name="Model", 
                           top_n=None, save=False, save_dir='reports/figures'):
    """
    Parameters:
    -----------
    model : sklearn model
        Modele entraine (doit avoir feature_importances_)
    feature_names : list
        Noms des variables
    model_name : str
        Nom du modele
    top_n : int
        Afficher les top_n features (None = toutes)
    save : bool
        Sauvegarder la figure
    save_dir : str
        Dossier de sauvegarde
    """
    if hasattr(model, 'named_steps'):
        classifier = model.named_steps['classifier']
    else:
        classifier = model
    
    if not hasattr(classifier, 'feature_importances_'):
        print(f"{model_name} ne fournit pas d'importance des features")
        return
    
    importances = classifier.feature_importances_
    
    importance_df = pd.DataFrame({
        'Variable': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    if top_n:
        importance_df = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Variable'], importance_df['Importance'], 
             color='skyblue', edgecolor='black')
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Importance des Variables - {model_name}', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    if save:
        os.makedirs(save_dir, exist_ok=True)
        filename = f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardee: {filepath}")
    
    plt.show()
    
    print("\n" + "=" * 50)
    print(f"IMPORTANCE DES VARIABLES - {model_name}")
    print("=" * 50)
    print(importance_df.to_string(index=False))

# ============================================================================
def afficher_barres_comparaison(df_results, save=False, save_dir='reports/figures'):
    """
    Affiche un graphique à barres comparant les modeles.
    
    Parameters:
    -----------
    df_results : pd.DataFrame
        Resultats de comparer_modeles()
    save : bool
        Sauvegarder la figure
    save_dir : str
        Dossier de sauvegarde
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    df_plot = df_results.set_index('Modele')[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']]
    df_plot.plot(kind='bar', ax=ax, rot=45)
    
    ax.set_title('Comparaison des Performances des Modeles', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score')
    ax.set_ylim([0.5, 1.0])
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, 'models_comparison_bars.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Figure sauvegardee: {filepath}")
    
    plt.show()

# ============================================================================
def generer_rapport(models_dict, X_test, y_test, feature_names, 
                   save_figures=True, save_dir='reports/figures'):
    """
    Parameters:
    -----------
    models_dict : dict
        Dictionnaire {nom_modele: modele}
    X_test : pd.DataFrame
        Variables de test
    y_test : pd.Series
        Cible de test
    feature_names : list
        Noms des variables
    save_figures : bool
        Sauvegarder les figures
    save_dir : str
        Dossier de sauvegarde
    
    Returns:
    --------
    dict
        Dictionnaire contenant le rapport complet
    """
    print("\n" + "=" * 70)
    print("GENERATION DU RAPPORT COMPLET D'EVALUATION")
    print("=" * 70)
    
    df_results = comparer_modeles(models_dict, X_test, y_test)
    
    print("\nGeneration des graphiques...")
    
    afficher_barres_comparaison(df_results, save=save_figures, save_dir=save_dir)
    
    afficher_courbes_roc(models_dict, X_test, y_test, save=save_figures, save_dir=save_dir)
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        afficher_matrice_confusion(y_test, y_pred, name, save=save_figures, save_dir=save_dir)
    
    for name, model in models_dict.items():
        if 'Random Forest' in name or 'Gradient Boosting' in name:
            afficher_importance_features(
                model, feature_names, name, 
                save=save_figures, save_dir=save_dir
            )
    
    print("\n" + "=" * 70)
    print("RAPPORT COMPLET GENERE")
    print("=" * 70)
    
    return {
        'results_table': df_results,
        'best_model': df_results.iloc[0]['Modele']
    }

# ============================================================================
def predire_nouvel_echantillon(model, sample_data, feature_names):
    """
    Parameters:
    -----------
    model : sklearn model
        Modele entraine
    sample_data : list ou dict
        Donnees de l'echantillon
    feature_names : list
        Noms des variables dans l'ordre
    
    Returns:
    --------
    dict
        Prediction et probabilite
    """
    if isinstance(sample_data, dict):
        df_sample = pd.DataFrame([sample_data])
    else:
        df_sample = pd.DataFrame([sample_data], columns=feature_names)
    
    prediction = model.predict(df_sample)[0]
    probability = model.predict_proba(df_sample)[0, 1]
    
    result = {
        'prediction': int(prediction),
        'prediction_label': 'Diabete' if prediction == 1 else 'Pas de diabete',
        'probability': probability
    }
    
    print("=" * 50)
    print("PREDICTION")
    print("=" * 50)
    print(f"Resultat: {result['prediction_label']}")
    print(f"Probabilite de diabete: {result['probability']:.2%}")
    print("=" * 50)
    
    return result

if __name__ == "__main__":
    from preprocess import pipeline_pretraitement
    from train import charger_modele
    
    data = pipeline_pretraitement()
    
    models = {
        'Random Forest (Tuned)': charger_modele('models/random_forest_tuned.pkl'),
        'Gradient Boosting (Tuned)': charger_modele('models/gradient_boosting_tuned.pkl')
    }
    
    report = generer_rapport(
        models,
        data['X_test'],
        data['y_test'],
        data['X_test'].columns.tolist(),
        save_figures=True
    )
    
    print("\nEvaluation terminee!")
    print(f"   Meilleur modele: {report['best_model']}")
