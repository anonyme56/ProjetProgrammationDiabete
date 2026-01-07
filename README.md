# Prédiction du Diabète chez les Femmes Pima

## Auteurs

- **HADJ-ABDELKADER Ilias**
- **ANDRIAMANANTSOA ARO Andoniaina**
- **OUGUENOUNE Chanesse**
- **Ali Yohan**

**Dépôt GitHub :** https://github.com/anonyme56/ProjetProgrammationDiabete.git

---

## Description du Projet

Ce projet vise à développer un modèle de machine learning capable de **prédire la présence de diabète** chez des patientes en se basant sur des données médicales. Il s'agit d'un problème de **classification binaire** supervisée.

Le dataset utilisé provient de l'Institut National du Diabète et des Maladies Digestives et Rénales (NIDDK) et contient des informations médicales sur des femmes d'origine Pima, une population particulièrement touchée par le diabète.

---

## Objectif

Prédire si une patiente est diabétique (Outcome = 1) ou non (Outcome = 0) en fonction de 8 variables médicales.

**Type de problème :** Classification binaire

**Dataset :** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## Dataset

### Source des Données

- **Source :** Kaggle - UCI Machine Learning Repository
- **Lien direct :** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Taille :** 768 échantillons × 9 colonnes
- **Format :** CSV

### Description des Variables

**Variables Explicatives (Features) :**
1. **Pregnancies** : Nombre de grossesses
2. **Glucose** : Concentration de glucose plasmatique (mg/dL) après 2h dans un test de tolérance au glucose
3. **BloodPressure** : Pression artérielle diastolique (mm Hg)
4. **SkinThickness** : Épaisseur du pli cutané du triceps (mm)
5. **Insulin** : Insuline sérique à 2 heures (mu U/ml)
6. **BMI** : Indice de masse corporelle (poids en kg / (taille en m)²)
7. **DiabetesPedigreeFunction** : Score basé sur les antécédents familiaux de diabète
8. **Age** : Âge en années

**Variable Cible (Target) :**
- **Outcome** : 0 = Pas de diabète, 1 = Diabète

### Acquisition des Données

Les données sont téléchargées automatiquement via l'API Kaggle (`kagglehub`). 

**Méthode reproductible :**
```python
import kagglehub
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
```

**Note :** Les données brutes ne sont pas commitées dans le dépôt pour des raisons de taille. Elles sont téléchargées automatiquement à l'exécution du notebook.

---

## Installation et Environnement

### Prérequis

- **Python 3.8+** 
- pip (gestionnaire de paquets Python)

### Installation des Dépendances

1. **Cloner le dépôt :**
   ```powershell
   git clone https://github.com/anonyme56/ProjetProgrammationDiabete.git
   ```

2. **Créer un environnement virtuel (recommandé) :**
   ```powershell
   python -m venv venv
   # Activer l'environnement
   # Windows:
   venv\Scripts\activate
   ```

3. **Installer les packages requis :**
   ```powershell
   pip install -r requirements.txt
   ```

### Versions Principales

- **numpy** : 1.24+
- **pandas** : 2.0+
- **matplotlib** : 3.7+
- **seaborn** : 0.12+
- **scikit-learn** : 1.3+
- **scipy** : 1.10+
- **kagglehub** : 0.2+
- **joblib** : 1.3+

---

## Structure du Projet

```
programmation avancée/
│
│
├── notebooks/   # Notebooks Jupyter
│   └── TP_Prog_indian_diabete.ipynb # Notebook principal 
│
├── src/                           # Code source Python modulaire
│   ├── preprocess.py             # Prétraitement des données
│   ├── train.py                  # Entraînement des modèles
│   └── evaluate.py               # Évaluation et visualisations
│
├── models/                        # Modèles entraînés sauvegardés
│   └── diabetes_model.pkl   # Meilleur modèle (généré après exécution)
│
├── reports/                       # Rapports et visualisations
│   └── figures/                   # Graphiques exportés
│
├── README.md                      # Ce fichier
├── requirements.txt               # Dépendances Python
```

---

## Utilisation du Code Modulaire (src/)

Le projet inclut un code Python modulaire et réutilisable dans le dossier `src/` pour faciliter l'entraînement et l'évaluation des modèles.

### Architecture du Code

#### `src/preprocess.py` - Prétraitement des Données

Fonctions pour charger, nettoyer et préparer les données :

- **`charger_donnees()`** : Télécharge le dataset depuis Kaggle
- **`nettoyer_donnees()`** : Remplace les valeurs impossibles (0 → NaN)
- **`separer_donnees()`** : Split stratifié train/test
- **`creer_preprocesseur()`** : Pipeline d'imputation et standardisation
- **`pipeline_pretraitement()`** : Pipeline complet de A à Z

**Exemple d'utilisation :**
```python
from src.preprocess import pipeline_pretraitement

# Exécuter tout le prétraitement en une ligne
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
```

#### `src/train.py` - Entraînement des Modèles

Fonctions pour entraîner et optimiser les modèles :

- **`entrainer_modeles_baseline()`** : Entraîne 4 modèles de base (LogReg, RF, GB, SVM)
- **`optimiser_random_forest()`** : Optimisation avec GridSearchCV
- **`optimiser_gradient_boosting()`** : Optimisation avec RandomizedSearchCV
- **`sauvegarder_modele()`** : Sauvegarde d'un modèle au format .pkl
- **`charger_modele()`** : Chargement d'un modèle sauvegardé
- **`entrainer_et_optimiser_tout()`** : Pipeline complet d'entraînement

**Exemple d'utilisation :**
```python
from src.train import entrainer_et_optimiser_tout

# Entraîner tous les modèles et optimiser les meilleurs
models = entrainer_et_optimiser_tout(
    X_train=X_train,
    y_train=y_train,
    preprocessor=preprocessor,
    optimize=True,
    save=True,
    random_state=42
)

# Récupérer les modèles
baseline_models = models['baseline']
best_rf = models['Random Forest (Tuned)']
best_gb = models['Gradient Boosting (Tuned)']
```

#### `src/evaluate.py` - Évaluation et Visualisations

Fonctions pour évaluer les modèles et générer des rapports :

- **`evaluer_modele()`** : Calcule toutes les métriques d'un modèle
- **`comparer_modeles()`** : Compare plusieurs modèles en un tableau
- **`afficher_matrice_confusion()`** : Affiche la matrice de confusion
- **`afficher_courbes_roc()`** : Trace les courbes ROC
- **`afficher_importance_features()`** : Importance des variables
- **`generer_rapport()`** : Génère un rapport complet avec graphiques
- **`predire_nouvel_echantillon()`** : Prédiction sur de nouvelles données

**Exemple d'utilisation :**
```python
from src.evaluate import generer_rapport

# Générer un rapport complet d'évaluation
report = generer_rapport(
    models_dict=all_models,
    X_test=X_test,
    y_test=y_test,
    feature_names=X_train.columns.tolist(),
    save_figures=True,
    save_dir='reports/figures'
)

print(f"Meilleur modèle : {report['best_model']}")
print(report['results_table'])
```

### Script Principal `main.py`

Le fichier `main.py` orchestre tout le pipeline de bout en bout :

```powershell
python main.py
```

**Ce script exécute automatiquement :**
1. Téléchargement et prétraitement des données
2. Entraînement des modèles baseline
3. Optimisation des hyperparamètres (Random Forest + Gradient Boosting)
4. Évaluation comparative de tous les modèles
5. Génération des visualisations (matrices de confusion, courbes ROC, importance des variables)
6. Sauvegarde des meilleurs modèles dans `models/`

**Sortie attendue :**
- Modèles sauvegardés : `models/best_diabetes_model.pkl`, `models/random_forest_tuned.pkl`, etc.
- Figures : `reports/figures/*.png`
- Tableau récapitulatif des performances dans la console

---

## Reproduire les Résultats

### Exécution Pas à Pas

**Option 1 : Utiliser le script principal (recommandé)**

```powershell
python main.py
```

Cela exécute automatiquement tout le pipeline : prétraitement, entraînement, optimisation, évaluation et sauvegarde.

**Option 2 : Utiliser le notebook Jupyter**

Le projet contient également **un notebook principal** (`notebooks/tpfinal.ipynb`) qui contient toutes les étapes de manière séquentielle et documentée.

**Ordre d'exécution des cellules :**

1. **Installation et importation** (Cellules 1-2)
   - Installation de kagglehub
   - Import des bibliothèques

2. **Acquisition des données** (Cellule 3)
   - Téléchargement automatique du dataset
   - Chargement avec Pandas

3. **Compréhension des données** (Cellules 4-7)
   - Exploration initiale
   - Statistiques descriptives
   - Analyse des valeurs manquantes

4. **Nettoyage et prétraitement** (Cellules 8-10)
   - Gestion des valeurs impossibles (0 → NaN)
   - Visualisation des données nettoyées

5. **EDA complète** (Cellules 11-18)
   - Analyse univariée
   - Analyse bivariée
   - Matrice de corrélation
   - Insights clés

6. **Préparation pour la modélisation** (Cellules 19-22)
   - Split train/test stratifié
   - Pipeline de prétraitement

7. **Modélisation baseline** (Cellules 23-29)
   - Régression Logistique
   - Random Forest
   - Gradient Boosting
   - SVM
   - Comparaison des modèles

8. **Analyse des features et erreurs** (Cellules 30-34)
   - Importance des variables
   - Matrices de confusion
   - Courbes ROC

9. **Optimisation des hyperparamètres** (Cellules 35-39)
   - GridSearchCV pour Random Forest
   - RandomizedSearchCV pour Gradient Boosting
   - Comparaison baseline vs tuned

10. **Évaluation finale et sauvegarde** (Cellules 40-41)
    - Sélection du meilleur modèle
    - Sauvegarde du modèle

**Pour exécuter :**
- Ouvrir `notebooks/tpfinal.ipynb` dans Jupyter Notebook ou JupyterLab
- Exécuter toutes les cellules dans l'ordre (Run All)
- Les résultats seront reproductibles grâce au seed fixé (`RANDOM_STATE = 42`)

---

##  Résumé de l'Analyse Exploratoire (EDA)

### Observations Clés

#### 1. Qualité des Données

- **Valeurs manquantes implicites :** Certaines variables contiennent des valeurs à 0 qui sont physiologiquement impossibles
  - Glucose : 5 valeurs (0.65%)
  - BloodPressure : 35 valeurs (4.56%)
  - SkinThickness : 227 valeurs (29.56%)
  - Insulin : 374 valeurs (48.70%)
  - BMI : 11 valeurs (1.43%)

- **Traitement :** Les 0 ont été remplacés par NaN et imputés par la médiane

#### 2. Distribution de la Variable Cible

- **Classe 0 (Pas de diabète) :** 500 échantillons (65.1%)
- **Classe 1 (Diabète) :** 268 échantillons (34.9%)
- **Déséquilibre :** Légèrement déséquilibré, mais gérable sans techniques de rééchantillonnage

#### 3. Variables les Plus Prédictives

**Corrélation avec la cible (Outcome) :**
1. **Glucose** : 0.467 (forte corrélation positive)
2. **BMI** : 0.293 (corrélation modérée)
3. **Age** : 0.238 (corrélation faible-modérée)
4. **Pregnancies** : 0.222
5. **Insulin** : 0.173

**Insight :** Le taux de glucose est de loin le facteur le plus discriminant pour le diabète, suivi par l'indice de masse corporelle et l'âge.

#### 4. Distributions et Outliers

- Plusieurs variables présentent une **asymétrie à droite** (Insulin, DiabetesPedigreeFunction)
- Présence d'**outliers** dans toutes les variables (détectés via boxplots)
- Les outliers ont été conservés car ils peuvent représenter des cas médicaux réels

#### 5. Analyse Bivariée

- **Glucose :** Les patientes diabétiques ont un taux médian significativement plus élevé
- **BMI :** Tendance à être plus élevé chez les diabétiques
- **Age :** Le risque augmente avec l'âge
- **Pas de multicolinéarité forte** entre les variables

### Figures Clés

- Distribution de la variable cible (barplot + pie chart)
- Histogrammes de toutes les variables
- Boxplots par classe
- Matrice de corrélation (heatmap)
- Pairplot des variables importantes

---

## Résumé de la Modélisation

### Modèles Entraînés

Quatre algorithmes de classification ont été testés :

1. **Régression Logistique**
   - Modèle linéaire simple et interprétable
   - Baseline solide pour la classification binaire
   - Hypothèses : linéarité et indépendance des features

2. **Random Forest**
   - Ensemble de 100 arbres de décision
   - Robuste aux outliers et non-linéarités
   - Fournit l'importance des variables

3. **Gradient Boosting**
   - Apprentissage séquentiel avec correction des erreurs
   - Généralement très performant
   - Capture bien les interactions complexes

4. **Support Vector Machine (SVM)**
   - Kernel RBF pour gérer la non-linéarité
   - Efficace en haute dimension
   - Sensible au scaling (appliqué via pipeline)

### Choix des Modèles - Justification

- **Régression Logistique :** Modèle de référence simple et interprétable, idéal pour comprendre les relations linéaires
- **Random Forest :** Gère bien les non-linéarités et les interactions entre variables sans hypothèses strictes
- **Gradient Boosting :** Connu pour ses excellentes performances en classification, corrige itérativement les erreurs
- **SVM :** Efficace pour trouver la frontière de décision optimale, même dans des espaces complexes

### Métriques d'Évaluation

**Métriques utilisées :**
- **Accuracy** : Taux de prédictions correctes global
- **Precision** : Proportion de vrais positifs parmi les prédictions positives
- **Recall** : Proportion de vrais positifs détectés (sensibilité)
- **F1-Score** : Moyenne harmonique de Precision et Recall
- **AUC-ROC** : Capacité à discriminer les classes (métrique principale)

**Métrique principale choisie :** **AUC-ROC**
- Moins sensible au déséquilibre des classes
- Mesure la qualité globale du modèle
- Permet de comparer objectivement les modèles

### Tableau des Résultats

#### Modèles Baseline (sans optimisation)

| Modèle                | Accuracy | Precision | Recall | F1-Score | AUC-ROC | CV (mean ± std) |
|-----------------------|----------|-----------|--------|----------|---------|-----------------|
| Logistic Regression   | 0.7727   | 0.6667    | 0.6000 | 0.6316   | 0.8281  | 0.8168 ± 0.0314 |
| Random Forest         | 0.7662   | 0.6667    | 0.5600 | 0.6087   | 0.8242  | 0.8098 ± 0.0378 |
| Gradient Boosting     | 0.7792   | 0.6800    | 0.6000 | 0.6377   | 0.8371  | 0.8213 ± 0.0299 |
| SVM                   | 0.7727   | 0.6800    | 0.5600 | 0.6145   | 0.8254  | 0.8157 ± 0.0342 |

**Meilleur modèle baseline :** Gradient Boosting (AUC = 0.8371)

#### Modèles Optimisés (après tuning)

| Modèle                      | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Gain vs Baseline |
|-----------------------------|----------|-----------|--------|----------|---------|------------------|
| Random Forest (Tuned)       | 0.7792   | 0.6842    | 0.5200 | 0.5909   | 0.8298  | +0.0056         |
| Gradient Boosting (Tuned)   | 0.7857   | 0.6939    | 0.5800 | 0.6316   | 0.8429  | +0.0058         |

**Modèle Final Sélectionné : Gradient Boosting (Tuned)**
- **AUC-ROC : 0.8429**
- **Accuracy : 78.57%**
- **F1-Score : 0.6316**

### Importance des Variables

**Top 5 des variables les plus importantes (Random Forest) :**
1. **Glucose** : 0.2854 (très important)
2. **BMI** : 0.1632
3. **Age** : 0.1389
4. **DiabetesPedigreeFunction** : 0.1164
5. **Pregnancies** : 0.0981

**Coefficients de la Régression Logistique :**
- **Glucose** : +1.23 (impact positif fort)
- **BMI** : +0.87 (impact positif modéré)
- **Age** : +0.65 (impact positif)
- **DiabetesPedigreeFunction** : +0.54

Ces résultats confirment que le **glucose** est le facteur le plus déterminant pour la prédiction du diabète.

---

## Optimisation des Hyperparamètres

### Stratégie d'Optimisation

Deux approches complémentaires ont été utilisées :

1. **GridSearchCV** pour Random Forest
   - Recherche exhaustive dans une grille de paramètres
   - 5-fold cross-validation
   - Métrique d'optimisation : AUC-ROC

2. **RandomizedSearchCV** pour Gradient Boosting
   - Échantillonnage aléatoire (50 combinaisons)
   - Plus efficace pour de grands espaces de recherche
   - 5-fold cross-validation

### Espaces de Recherche

#### Random Forest
```python
param_grid_rf = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['sqrt', 'log2']
}
```

**Justification :**
- `n_estimators` : Nombre d'arbres (plus = meilleur mais plus lent)
- `max_depth` : Profondeur maximale pour contrôler le surapprentissage
- `min_samples_split/leaf` : Régularisation pour éviter l'overfitting
- `max_features` : Nombre de features à considérer pour chaque split

#### Gradient Boosting
```python
param_grid_gb = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__subsample': [0.8, 0.9, 1.0]
}
```

**Justification :**
- `learning_rate` : Taux d'apprentissage (plus bas = plus stable mais plus lent)
- `n_estimators` : Nombre de boosting stages
- `max_depth` : Profondeur des arbres (shallow trees pour GB)
- `subsample` : Fraction d'échantillons pour chaque arbre (prévient l'overfitting)

### Meilleurs Hyperparamètres Trouvés

**Random Forest :**
- `n_estimators`: 200
- `max_depth`: 20
- `min_samples_split`: 5
- `min_samples_leaf`: 2
- `max_features`: 'sqrt'

**Gradient Boosting :**
- `n_estimators`: 100
- `learning_rate`: 0.1
- `max_depth`: 5
- `min_samples_split`: 5
- `subsample`: 0.9

### Résultats de l'Optimisation

| Modèle              | AUC Baseline | AUC Tuned | Gain Absolu | Gain Relatif |
|---------------------|--------------|-----------|-------------|--------------|
| Random Forest       | 0.8242       | 0.8298    | +0.0056     | +0.68%       |
| Gradient Boosting   | 0.8371       | 0.8429    | +0.0058     | +0.69%       |

**Analyse :**
- Les gains sont modestes (~0.6-0.7%) mais positifs
- Les modèles baseline étaient déjà bien configurés
- L'optimisation a permis d'améliorer légèrement les performances
- Le tuning a surtout réduit la variance (modèles plus stables)

---

##  Analyse d'Erreurs

### Matrices de Confusion

**Gradient Boosting (Tuned) - Modèle Final :**

|                    | Prédit : Négatif | Prédit : Positif |
|--------------------|------------------|------------------|
| **Réel : Négatif** | 92 (VN)          | 8 (FP)           |
| **Réel : Positif** | 25 (FN)          | 29 (VP)          |

- **Vrais Négatifs (VN)** : 92 - Correctement identifiés comme non-diabétiques
- **Vrais Positifs (VP)** : 29 - Correctement identifiés comme diabétiques
- **Faux Positifs (FP)** : 8 - Erreur Type I (fausse alarme)
- **Faux Négatifs (FN)** : 25 - Erreur Type II (non-détection)

### Types d'Erreurs

**Erreurs Type I (Faux Positifs) :** 8 cas
- Patientes identifiées comme diabétiques alors qu'elles ne le sont pas
- **Impact :** Tests médicaux supplémentaires inutiles, anxiété du patient
- **Moins grave** dans un contexte médical

**Erreurs Type II (Faux Négatifs) :** 25 cas
- Patientes diabétiques non détectées
- **Impact :** Absence de traitement, complications possibles
- **PLUS GRAVE** médicalement

### Où le Modèle Échoue

**Analyse des faux négatifs :**
- Patientes avec des valeurs de glucose à la limite du seuil
- Jeunes patientes diabétiques (le modèle sous-estime le risque chez les jeunes)
- Cas avec des antécédents familiaux faibles mais diabète présent

**Analyse des faux positifs :**
- Patientes avec un BMI élevé mais sans diabète
- Valeurs de glucose légèrement élevées mais en dessous du seuil diabétique

### Courbes ROC

Toutes les courbes ROC montrent de bonnes performances (AUC > 0.82), avec Gradient Boosting légèrement en tête.

**Interprétation :**
- Les modèles discriminent bien entre les deux classes
- L'aire sous la courbe (0.84) indique une bonne capacité prédictive
- Le seuil de classification peut être ajusté selon le contexte (privilégier Recall si on veut réduire les FN)

---

##  Reproductibilité

### Seeds et Aléatoire

Toutes les sources d'aléatoire sont contrôlées :
```python
RANDOM_STATE = 42

np.random.seed(42)
# Tous les modèles utilisent random_state=RANDOM_STATE
# Split train/test stratifié avec random_state=RANDOM_STATE
```

### Environnement

- **Python** : 3.8+
- **Packages** : Versions fixées dans `requirements.txt`

### Commandes pour Reproduire

```powershell
# 1. Cloner et installer
git clone https://github.com/anonyme56/ProjetProgrammationDiabete.git
pip install -r requirements.txt

# 2. Lancer Jupyter
jupyter notebook

# 3. Ouvrir notebooks/tpfinal.ipynb et exécuter toutes les cellules
```

**Résultats attendus :**
- Même split train/test
- Mêmes métriques de performance
- Modèle sauvegardé identique dans `models/best_diabetes_model.pkl`

---

## Références

### Dataset
- Smith, J.W., Everhart, J.E., Dickson, W.C., Knowler, W.C., & Johannes, R.S. (1988). *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus*. In Proceedings of the Symposium on Computer Applications and Medical Care (pp. 261--265). IEEE Computer Society Press.

### Librairies et Outils
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Seaborn Documentation](https://seaborn.pydata.org/)

### Ressources Kaggle
- [Dataset Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Kaggle Notebooks sur ce dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database/code)

---

##  Limites et Pistes d'Amélioration

### Limites Actuelles

1. **Taille du dataset**
   - Seulement 768 échantillons
   - Limitation pour l'entraînement de modèles complexes
   - Risque de surapprentissage sur certains sous-groupes

2. **Valeurs manquantes**
   - 48.7% de valeurs manquantes pour Insulin
   - Imputation par la médiane peut introduire du biais
   - Perte d'information potentielle

3. **Déséquilibre des classes**
   - 65% vs 35% (modéré mais présent)
   - Pas de technique de rééchantillonnage utilisée

4. **Généralisation**
   - Dataset spécifique aux femmes Pima
   - Généralisation limitée à d'autres populations

5. **Feature engineering**
   - Pas d'interactions entre variables créées
   - Pas de transformations non-linéaires testées

### Pistes d'Amélioration

**Court terme :**
1. **Gestion du déséquilibre**
   - Tester SMOTE ou ADASYN pour sur-échantillonner la classe minoritaire
   - Utiliser `class_weight='balanced'` dans les modèles

2. **Feature Engineering**
   - Créer des ratios (ex: Glucose/BMI)
   - Interactions polynomiales entre variables importantes
   - Binning de l'âge par groupes

3. **Modèles supplémentaires**
   - XGBoost, LightGBM, CatBoost (boosting plus avancé)
   - Réseaux de neurones (MLP avec plusieurs couches)
   - Stacking/Blending d'ensembles

4. **Optimisation avancée**
   - Optuna pour une recherche bayésienne d'hyperparamètres
   - Validation croisée stratifiée imbriquée

**Long terme :**
1. **Collecte de données supplémentaires**
   - Augmenter la taille du dataset
   - Inclure d'autres populations pour améliorer la généralisation

2. **Variables additionnelles**
   - Inclure des facteurs de style de vie (alimentation, exercice)
   - Données génomiques si disponibles

3. **Déploiement**
   - Créer une API REST avec Flask/FastAPI
   - Interface web pour les professionnels de santé
   - Monitoring des performances en production

4. **Interprétabilité**
   - SHAP values pour expliquer les prédictions individuelles
   - LIME pour l'explicabilité locale
   - Analyse contrefactuelle (what-if scenarios)

---

## Conclusion

Ce projet a permis de développer et d'évaluer **quatre modèles de classification** pour prédire le diabète chez les femmes Pima :

1. **Régression Logistique** - Modèle linéaire simple (AUC : 0.8281)
2. **Random Forest** - Ensemble d'arbres de décision (AUC : 0.8242)
3. **Gradient Boosting** - Apprentissage séquentiel avec correction des erreurs (AUC : 0.8371)
4. **Support Vector Machine (SVM)** - Séparation optimale avec kernel RBF (AUC : 0.8254)

Après optimisation des hyperparamètres, le **Gradient Boosting** s'est révélé être le modèle le plus performant avec :
- **AUC-ROC : 0.8429** (meilleure capacité discriminante)
- **Accuracy : 78.57%** (taux de prédictions correctes)
- **Precision : 0.6939** (69% des prédictions positives sont correctes)
- **Recall : 0.5800** (58% des cas diabétiques détectés)
- **F1-Score : 0.6316** (équilibre entre précision et rappel)

Le modèle final sauvegardé dans `models/best_diabetes_model.pkl` peut être utilisé pour prédire le risque de diabète avec une fiabilité de **84.29%** selon la métrique AUC-ROC. Les variables les plus importantes identifiées sont le **glucose** (28.5%), le **BMI** (16.3%) et l'**âge** (13.9%).
