import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess import pipeline_pretraitement
from src.train import entrainer_et_optimiser_tout
from src.evaluate import generer_rapport

import warnings
warnings.filterwarnings('ignore')


def main():
    """
    Fonction principale qui exécute le pipeline complet.
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "PIPELINE COMPLET DE MACHINE LEARNING")
    print(" " * 25 + "Prédiction du Diabète")
    print("=" * 80)
    
    # ==================================================================
    # ÉTAPE 1: PRÉTRAITEMENT DES DONNÉES
    # ==================================================================
    print("\n" + "_" * 40)
    print("ÉTAPE 1/4: PRÉTRAITEMENT DES DONNÉES")
    print("_" * 40)
    
    data = pipeline_pretraitement(
        dataset_name="uciml/pima-indians-diabetes-database",
        test_size=0.2,
        impute_strategy='median',
        random_state=42
    )
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    preprocessor = data['preprocessor']
    
    print("\n[OK] Prétraitement terminé avec succès!")
    
    # ==================================================================
    # ÉTAPE 2: ENTRAÎNEMENT ET OPTIMISATION
    # ==================================================================
    print("\n" + "_" * 40)
    print("ÉTAPE 2/4: ENTRAÎNEMENT ET OPTIMISATION DES MODÈLES")
    print("_" * 40)
    
    models = entrainer_et_optimiser_tout(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        optimize=True,
        save=True,
        random_state=42
    )
    
    print("\n[OK] Entraînement et optimisation terminés!")
    
    # ==================================================================
    # ÉTAPE 3: ÉVALUATION ET COMPARAISON
    # ==================================================================
    print("\n" + "_" * 40)
    print("ÉTAPE 3/4: ÉVALUATION ET COMPARAISON DES MODÈLES")
    print("_" * 40)
    
    # Préparer le dictionnaire de tous les modèles pour la comparaison
    all_models = {
        **models['baseline'],
        'Random Forest (Tuned)': models['Random Forest (Tuned)'],
        'Gradient Boosting (Tuned)': models['Gradient Boosting (Tuned)']
    }
    
    report = generer_rapport(
        models_dict=all_models,
        X_test=X_test,
        y_test=y_test,
        feature_names=X_train.columns.tolist(),
        save_figures=True,
        save_dir='reports/figures'
    )
    
    print("\n[OK] Évaluation terminée!")
    
    # ==================================================================
    # ÉTAPE 4: RÉSUMÉ FINAL
    # ==================================================================
    print("\n" + "_" * 40)
    print("ÉTAPE 4/4: RÉSUMÉ FINAL")
    print("_" * 40)
    
    print("\nTABLEAU RÉCAPITULATIF:")
    print(report['results_table'].to_string(index=False))
    
    print(f"\nMEILLEUR MODÈLE: {report['best_model']}")
    
    best_metrics = report['results_table'].iloc[0]
    print(f"\nPERFORMANCES DU MEILLEUR MODÈLE:")
    print(f"   - Accuracy:  {best_metrics['Accuracy']:.4f}")
    print(f"   - Precision: {best_metrics['Precision']:.4f}")
    print(f"   - Recall:    {best_metrics['Recall']:.4f}")
    print(f"   - F1-Score:  {best_metrics['F1-Score']:.4f}")
    print(f"   - AUC-ROC:   {best_metrics['AUC-ROC']:.4f}")
    
    print("\n" + "=" * 80)
    print("[OK] PIPELINE COMPLET TERMINÉ AVEC SUCCÈS!")
    print("=" * 80)
    
    print("\nFichiers générés:")
    print("   - models/best_diabetes_model.pkl")
    print("   - models/random_forest_tuned.pkl")
    print("   - models/gradient_boosting_tuned.pkl")
    print("   - reports/figures/*.png")
    
    print("\nPour utiliser le modele:")
    print("   >>> from src.train import charger_modele")
    print("   >>> model = charger_modele('models/best_diabetes_model.pkl')")
    print("   >>> predictions = model.predict(new_data)")
    
    print("\n" + "=" * 80)
    
    return report


if __name__ == "__main__":
    try:
        report = main()
        print("\n[OK] Exécution réussie!")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERREUR] Erreur lors de l'exécution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
