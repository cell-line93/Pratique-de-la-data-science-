import pandas as pd
import glob
import ta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
import os
import shap


# -------------------------------------------
# 1.1 Préparation du dataset
# -------------------------------------------

def create_labeled_dataset(path: str) -> Dict[str, pd.DataFrame]:

    filepaths = glob.glob(f"{path}/*.csv")
    print(f"{len(filepaths)} fichiers trouvés dans {path}")
    dataset_dict: Dict[str, pd.DataFrame] = {}

    for file in filepaths:
        df = pd.read_csv(file)
        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0].split('_')[0]

        df = df[['Close']]
        df['Close Horizon'] = df['Close'].shift(-20)
        df['Horizon Return'] = (df['Close Horizon'] - df['Close']) / df['Close']
        df['Label'] = df['Horizon Return'].apply(lambda x: 2 if x > 0.05 else (0 if x < -0.05 else 1))

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
        
        # clear Na
        df = df.dropna(axis=0)
        
        dataset_dict[filename] = df
        
    return dataset_dict

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def prepare_data(dico: Dict[str, pd.DataFrame], use_smote=False):
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
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }
    model = SVC()
    best_model = tune_model(model, param_grid, X_train, y_train)
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
    return best_model

def evaluate_model(best_model, X_test, y_test, feature_names=None, model_type="tree", display_shap=True):
    """
    Évalue le modèle en affichant le rapport de classification, la précision et le graphique SHAP.
    
    Args:
        model: Le modèle entraîné.
        X_test (ndarray): Données de test.
        y_test (ndarray): Cibles de test.
        X_train (ndarray): Données d'entraînement pour SHAP.
    """
   
    preds = best_model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, preds))

    if not display_shap:
        return

    print("=== Interprétation SHAP ===")
    try:
        if model_type == "tree":
            explainer = shap.TreeExplainer(best_model)
            shap_values = explainer.shap_values(X_test)

        elif model_type == "kernel":
            explainer = shap.KernelExplainer(best_model.predict_proba, shap.kmeans(X_test, 10))
            shap_values = explainer.shap_values(X_test[:20])
            X_test = X_test.iloc[:20]


        else:
            raise ValueError("model_type must be 'tree' or 'kernel'")
        
        # Summary plot for class "buy" (2)
        shap.summary_plot(shap_values[:,:,2], X_test, feature_names=feature_names)

        # Summary plot for class "hold" (1)
        shap.summary_plot(shap_values[:,:,1], X_test, feature_names=feature_names)

        # Summary plot for class "sell" (0)
        shap.summary_plot(shap_values[:,:,0], X_test, feature_names=feature_names)

    except Exception as e:
        print("Erreur lors du calcul ou affichage des SHAP values :", e)


if __name__ == "__main__":
    dico = create_labeled_dataset(r"C:\Users\tagob\Documents\DAUPHINE\Pratique de la data science\TP\TP1\Companies_historical_data")
    X_train, X_test, y_train, y_test, feature_columns = prepare_data(dico, True)

    random_forest = train_random_forest(X_train,y_train)
    evaluate_model(random_forest,X_test,y_test,feature_columns,model_type="tree")