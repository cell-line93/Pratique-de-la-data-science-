import pandas as pd
import glob
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
import json
import joblib
import shap


# -------------------------------------------
# 1.1 Préparation du dataset
# -------------------------------------------

def create_labeled_dataset(path: str) -> Dict[str, pd.DataFrame]:

    filepaths = glob.glob(f"{path}/*.csv")
    print(f"{len(filepaths)} fichiers trouvés dans {path}")
    dataset_dict: Dict[str, pd.DataFrame] = {}
    dataset_val:  Dict[str, pd.DataFrame] = {}

    for file in filepaths:
        df = pd.read_csv(file)
        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0].split('_')[0]

        df = df[['Close']]
        df['Close Horizon'] = df['Close'].shift(-20)
        df['Horizon Return'] = (df['Close Horizon'] - df['Close']) / df['Close']
        

        # Indicateurs techniques
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'],window=20,fillna=True)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'],window=20,fillna=True)
        df['MACD'] = ta.trend.macd(df['Close'],window_slow=26,window_fast=12,fillna=True)
        df['MACD_Signal'] = ta.trend.macd_signal(df['Close'],window_slow=26,window_fast=12,fillna=True)
        df['RSI_14'] = ta.momentum.rsi(df['Close'],window=14,fillna=True)
        df['ROC_10'] = ta.momentum.roc(df['Close'],window=10,fillna=True)
        df['Bollinger_High'] = ta.volatility.bollinger_hband(df['Close'],window=20,window_dev=2,fillna=True)
        df['Bollinger_Low'] = ta.volatility.bollinger_lband(df['Close'],window=20,window_dev=2,fillna=True)
        df['Rolling_Volatility_20'] = df['Close'].pct_change().rolling(window=20).std()
        
        df_prevision = df[-20:]
        df = df[:-20]

        df['Label'] = df['Horizon Return'].apply(lambda x: 2 if x > 0.05 else (0 if x < -0.05 else 1))

        # clear Na
        df = df.dropna(axis=0)
        
        dataset_dict[filename] = df
        dataset_val[filename] = df_prevision
        
    return [dataset_dict,dataset_val]

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def prepare_data(dico: Dict[str, pd.DataFrame], use_smote=True):
    """
    Prépare les données pour le pipeline de classification.
    
    Args:
        dictionnaire des df par entreprise: Données avec les features et la colonne 'Label'.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """

    # Dataset global par concaténation
    dataset_global= pd.DataFrame()
    for key, value in dico.items():
        dataset_global = pd.concat([dataset_global,value],ignore_index=True)

    # Scinder en target et explicatives
    X = dataset_global.drop(columns=['Close Horizon','Horizon Return','Label'])  # Supprimer les colonnes non explicatives
    y = dataset_global['Label']
    
    feature_columns = X.columns

    # Standardisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Séparation train/test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)

    # Smote
    if use_smote:
        X_train, y_train = apply_smote(X_train, y_train)
    
    return X_train, X_test, y_train, y_test, feature_columns


# -------------------------------------------
# 1.2 Algorithmes de classification
# -------------------------------------------

def tune_model(model, param_grid, X_train, y_train):
    """
    Effectue un grid search pour déterminer les meilleurs hyperparamètres.
    
    Args:
        model: L'algorithme de classification à utiliser.
        param_grid (dict): Dictionnaire contenant les hyperparamètres à tester.
        X_train (ndarray): Données d'entraînement.
        y_train (ndarray): Cibles d'entraînement.
        
    Returns:
        model: Le modèle ajusté avec les meilleurs hyperparamètres.
    """
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def train_xgboost(X_train, y_train):
    """
    Applique XGBoost avec GridSearch pour ajuster les hyperparamètres.
    
    Args:
        X_train (ndarray): Données d'entraînement.
        y_train (ndarray): Cibles d'entraînement.
        
    Returns:
        XGBClassifier: Le modèle XGBoost entraîné.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    model = XGBClassifier()
    best_model = tune_model(model, param_grid, X_train, y_train)
    model_path = os.path.join("Projet Final", "Classification", "Fine-tuning", "xgboost_model.pkl")
    joblib.dump(best_model, model_path)
    return best_model


def train_random_forest(X_train, y_train):
    """
    Applique Random Forest avec GridSearch pour ajuster les hyperparamètres.
    
    Args:
        X_train (ndarray): Données d'entraînement.
        y_train (ndarray): Cibles d'entraînement.
        
    Returns:
        RandomForestClassifier: Le modèle Random Forest entraîné.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5]
    }
    model = RandomForestClassifier(random_state=42)
    best_model = tune_model(model, param_grid, X_train, y_train)
    model_path = os.path.join("Projet Final", "Classification", "Fine-tuning", "random_forest_model.pkl")
    joblib.dump(best_model, model_path)
    return best_model

def train_knn(X_train, y_train):
    """
    Applique KNN avec GridSearch pour ajuster les hyperparamètres.
    
    Args:
        X_train (ndarray): Données d'entraînement.
        y_train (ndarray): Cibles d'entraînement.
        
    Returns:
        KNeighborsClassifier: Le modèle KNN entraîné.
    """
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'metric': ['euclidean', 'manhattan']
    }
    model = KNeighborsClassifier()
    best_model = tune_model(model, param_grid, X_train, y_train)
    model_path = os.path.join("Projet Final", "Classification", "Fine-tuning", "knn_model.pkl")
    joblib.dump(best_model, model_path)
    return best_model



def train_svm(X_train, y_train):
    """
    Applique SVM avec GridSearch pour ajuster les hyperparamètres.
    
    Args:
        X_train (ndarray): Données d'entraînement.
        y_train (ndarray): Cibles d'entraînement.
        
    Returns:
        SVC: Le modèle SVM entraîné.
    """
    param_grid = {
        'C': [0.1, 1],
        'kernel': ['linear', 'rbf']
    }
    model = SVC(probability=True)
    best_model = tune_model(model, param_grid, X_train, y_train)
    model_path = os.path.join("Projet Final", "Classification", "Fine-tuning", "svm_model.pkl")
    joblib.dump(best_model, model_path)
    return best_model



def train_logistic_regression(X_train, y_train):
    """
    Applique la régression logistique avec GridSearch pour ajuster les hyperparamètres.
    
    Args:
        X_train (ndarray): Données d'entraînement.
        y_train (ndarray): Cibles d'entraînement.
        
    Returns:
        LogisticRegression: Le modèle de régression logistique entraîné.
    """
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear']
    }
    model = LogisticRegression()
    best_model = tune_model(model, param_grid, X_train, y_train)
    model_path = os.path.join("Projet Final", "Classification", "Fine-tuning", "reg_logistic_model.pkl")
    joblib.dump(best_model, model_path)
    return best_model



def evaluate_model(best_model,model_name, X_test, y_test, feature_names=None, model_type="tree", display_shap=False):
    """
    Évalue le modèle en affichant le rapport de classification, la précision,
    la matrice de confusion et les graphiques SHAP.
    
    Args:
        best_model: Le modèle entraîné.
        X_test (ndarray or DataFrame): Données de test.
        y_test (ndarray): Cibles de test.
        feature_names (list): Noms des variables explicatives.
        model_type (str): "tree" ou "kernel".
        display_shap (bool): Si True, affiche et enregistre les plots SHAP.
    """

    # Prédictions
    preds = best_model.predict(X_test)

    # === Rapport de classification ===
    report = classification_report(y_test, preds, output_dict=True)
    save_path = os.path.join("Projet Final", "Classification", "Evaluation",f"classification_report_{model_name}.json")

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    # === Matrice de confusion ===
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matrice de confusion")
    plt.savefig(os.path.join("Projet Final","Classification","Evaluation", f"confusion_matrix_{model_name}.png"))
    plt.close()

    if not display_shap:
        return report

    #=== Interprétation SHAP ===
    try:
        if model_type == "tree":
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test[:20])

        elif model_type == "kernel":
            explainer = shap.KernelExplainer(best_model.predict_proba, shap.kmeans(X_test[:20], 10))
            shap_values = explainer.shap_values(X_test[:20])
            

        else:
            raise ValueError("model_type must be 'tree' or 'kernel'")

        class_names = ["sell", "hold", "buy"]
        for i, class_name in enumerate(class_names):
            plt.title(f"SHAP Summary Plot - Classe {class_name}")
            shap.summary_plot(shap_values[i], X_test[:20], feature_names=feature_names, show=False)
            plt.savefig(os.path.join("Projet Final","Classification","Evaluation", f"shap_summary_{class_name}_{model_name}.png"))
            plt.close()

    except Exception as e:
        print("Erreur lors du calcul ou affichage des SHAP values :", e)

    return report 


def get_best_models(X_train, y_train):
    """
    Charge les modèles fine-tunés à partir de fichiers .pkl s'ils existent,
    sinon entraîne à nouveau les modèles et retourne la liste.

    Args:
        X_train (ndarray): Données d'entraînement.
        y_train (ndarray): Cibles d'entraînement.

    Returns:
        dictionnaire: dico des modèles entraînés ou chargés dans l'ordre :
              [XGBoost, RandomForest, KNN, LogisticRegression, SVM]
    """
    # Dossier de sauvegarde des modèles
    path = os.path.join("Projet Final", "Classification", "Fine-tuning")

    # Liste des fichiers .pkl présents
    filepaths = glob.glob(os.path.join(path, "*.pkl"))

    # Extraire les préfixes de nom de fichier (ex: "xgboost_model" depuis "xgboost_model.pkl")
    existing_models = [os.path.splitext(os.path.basename(file))[0] for file in filepaths]

    # Chargement ou entraînement des modèles selon leur présence sur disque
    xgboost_model = joblib.load(os.path.join(path, "xgboost_model.pkl")) if "xgboost_model" in existing_models else train_xgboost(X_train, y_train)
    random_forest_model = joblib.load(os.path.join(path, "random_forest_model.pkl")) if "random_forest_model" in existing_models else train_random_forest(X_train, y_train)
    knn_model = joblib.load(os.path.join(path, "knn_model.pkl")) if "knn_model" in existing_models else train_knn(X_train, y_train)
    reg_logistic_model = joblib.load(os.path.join(path, "reg_logistic_model.pkl")) if "reg_logistic_model" in existing_models else train_logistic_regression(X_train, y_train)
    svm_model = joblib.load(os.path.join(path, "svm_model.pkl")) if "svm_model" in existing_models else train_svm(X_train, y_train)

    dico_models = {
        "xgboost_model":xgboost_model, 
        "random_forest_model":random_forest_model, 
        "knn_model":knn_model, 
        "reg_logistic_model":reg_logistic_model,
        "svm_model":svm_model
    }
    return dico_models


def evaluate_all_models(models, X, y, feature_columns):
    best_model = None
    best_f1_score = -1
    best_model_name = ""

    for model_name, model in models.items():
        if model_name in ["xgboost_model", "random_forest_model"]:
            report = evaluate_model(model, model_name, X, y, feature_columns, model_type="tree")
        else:
            report = evaluate_model(model, model_name, X, y, feature_columns, model_type="kernel")
        
        # On suppose que evaluate_model retourne classification_report avec output_dict=True
        f1_score = report['1']['f1-score']  # ou 'weighted avg' si tu préfères
        
        if f1_score > best_f1_score:
            best_f1_score = f1_score
            best_model = model
            best_model_name = model_name

    return best_model_name, best_model, best_f1_score


def give_actual_recommendation_all_companies():
    
    dico = create_labeled_dataset("Projet Final\\Companies_historical_data")
    X_train, X_test, y_train, y_test, feature_columns = prepare_data(dico[0], True)
    dico_recommendation = {}

    models = get_best_models(X_train, y_train) #xgboost_model,random_forest_model,knn_model,reg_logistic_model,svm_model
    best_model_name, best_model, best_f1_score = evaluate_all_models(models,X_test,y_test,feature_columns)
    print("best model", best_model_name)
    print("f1 score du best model", best_f1_score)
    
    for name, df in dico[1].items():
        df = df.drop(columns=['Horizon Return', 'Close Horizon'])
        preds = best_model.predict(df)
        dico_recommendation[name] = (
            "sell" if preds[-1] == 0
            else "hold" if preds[-1] == 1
            else "buy"
        )

    save_path = os.path.join("Projet Final", "Classification", "Resultats","recommendation_companies.json")
    # Sauvegarde 
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dico_recommendation, f, ensure_ascii=False, indent=4)

    return dico_recommendation