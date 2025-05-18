import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from typing import Tuple, List, Dict
import os 
import json
import joblib


# -------------------------------------------
# 1.1 Création du dataset pour la régression
# -------------------------------------------

def create_target_features(df, n_days: int=30):
    """
        Crée les features et labels pour la régression.
        Chaque X[i] contient les n_days valeurs précédentes de la colonne "Close"
        Y[i] est la valeur à prédire : Close à J+1
    """
    x, y = [], []
    for i in range(n_days, len(df)):
        x.append(df[i-n_days:i, 0])
        y.append(df[i, 0])
    return np.array(x), np.array(y)

def prepare_data(filepath, n_days=30):
    """
        Charge un fichier CSV contenant les cours de bourse et prépare les données :
        - Extraction de "Close"
        - Normalisation
        - Création de features/target
        - Split train/test
    """
    df = pd.read_csv(filepath)
    df = df[["Close"]].dropna()
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    X, y = create_target_features(df_scaled, n_days=n_days)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    X_last_day = df_scaled[-n_days:].reshape(1, -1)
    return [X_train, X_test, y_train, y_test, scaler, X_last_day]



def data_for_all_company(path):
    """
        Applique la préparation des données à tous les fichiers CSV du dossier.
    """
    filepaths = glob.glob(f"{path}/*.csv")
    print(f"{len(filepaths)} fichiers trouvés dans {path}")
    data_dict: Dict[str, List] = {}

    for file in filepaths:
        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0].split('_')[0]
        data_dict[filename] = prepare_data(file)

    return data_dict

# -------------------------------------------
# 1.2 Algorithmes de régression
# -------------------------------------------

def tune_model(model, param_grid: Dict[str, List], X_train: np.ndarray, y_train: np.ndarray):
    """
        Effectue un GridSearch pour trouver les meilleurs hyperparamètres.
    """
    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

def train_xgboost(X_train, y_train):
    """
    Applique XGBoost avec GridSearch pour ajuster les hyperparamètres.
    
    Args:
        X_train (ndarray): Données d'entraînement.
        y_train (ndarray): Cibles d'entraînement.
        
    Returns:
        XGBRegressor: Le modèle XGBoost entraîné.
    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5],
        'learning_rate': [0.01, 0.1, 0.2]
    }
     
    model = XGBRegressor()
    best_model = tune_model(model, param_grid, X_train, y_train)
    model_path = os.path.join("Projet Final", "Regression", "ML","Fine-tuning", "xgboost_reg_model.pkl")
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
    model = RandomForestRegressor(random_state=42)
    best_model = tune_model(model, param_grid, X_train, y_train)
    model_path = os.path.join("Projet Final", "Regression", "ML","Fine-tuning", "random_forest_reg_model.pkl")
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
    model = KNeighborsRegressor()
    best_model = tune_model(model, param_grid, X_train, y_train)
    model_path = os.path.join("Projet Final", "Regression", "ML","Fine-tuning", "knn_reg_model.pkl")
    joblib.dump(best_model, model_path)
    return best_model

def train_linear_regression(X_train, y_train):
    """
    Applique la régression logistique avec GridSearch pour ajuster les hyperparamètres.
    
    Args:
        X_train (ndarray): Données d'entraînement.
        y_train (ndarray): Cibles d'entraînement.
        
    Returns:
        LinearRegression: Le modèle de régression lineaire entraîné.
    """
    param_grid = {
    }
    model = LinearRegression()
    best_model = tune_model(model, param_grid, X_train, y_train)
    model_path = os.path.join("Projet Final", "Regression", "ML","Fine-tuning", "reg_linear_model.pkl")
    joblib.dump(best_model, model_path)
    return best_model


def get_best_models(X_train, y_train):
    """
    Charge les modèles fine-tunés à partir de fichiers .pkl s'ils existent,
    sinon entraîne à nouveau les modèles et retourne la liste.

    Args:
        X_train (ndarray): Données d'entraînement.
        y_train (ndarray): Cibles d'entraînement.

    Returns:
        dictionnaire: dico des modèles entraînés ou chargés dans l'ordre :
              [XGBoost, RandomForest, KNN, LinearRegression]
    """
    # Dossier de sauvegarde des modèles
    path = os.path.join("Projet Final", "Regression", "ML","Fine-tuning")

    # Liste des fichiers .pkl présents
    filepaths = glob.glob(os.path.join(path, "*.pkl"))
    # Extraire les préfixes de nom de fichier (ex: "xgboost_model" depuis "xgboost_model.pkl")
    existing_models = [os.path.splitext(os.path.basename(file))[0] for file in filepaths]

    # Chargement ou entraînement des modèles selon leur présence sur disque
    xgboost_reg_model = joblib.load(os.path.join(path, "xgboost_reg_model.pkl")) if "xgboost_reg_model" in existing_models else train_xgboost(X_train, y_train)
    random_forest_reg_model = joblib.load(os.path.join(path, "random_forest_reg_model.pkl")) if "random_forest_reg_model" in existing_models else train_random_forest(X_train, y_train)
    knn_reg_model = joblib.load(os.path.join(path, "knn_reg_model.pkl")) if "knn_reg_model" in existing_models else train_knn(X_train, y_train)
    reg_linear_model = joblib.load(os.path.join(path, "reg_linear_model.pkl")) if "reg_linear_model" in existing_models else train_linear_regression(X_train, y_train)

    dico_models = {
        "xgboost_reg_model":xgboost_reg_model, 
        "random_forest_reg_model":random_forest_reg_model, 
        "knn_reg_model":knn_reg_model, 
        "reg_linear_model":reg_linear_model
    }
    return dico_models


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, scaler: MinMaxScaler) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
        Évalue un modèle avec les métriques MSE et RMSE après inversion du scaling.
    """
    preds = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1))
    mse = mean_squared_error(y_test_inv, preds_inv)
    rmse = np.sqrt(mse)
    return mse, rmse, preds_inv.flatten(), y_test_inv.flatten()

def plot_predictions(y_real_full: np.ndarray, y_train_len: int, preds: np.ndarray, label: str, name:str):
    """
        Affiche la courbe des vraies valeurs et des prédictions pour un modèle.
        Les prédictions sont décalées pour correspondre à leur position temporelle.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_real_full)), y_real_full, color='red', label='Valeurs réelles')
    plt.plot(range(y_train_len + 30, y_train_len + 30 + len(preds)),preds,label=label)
    plt.title(f"Prédictions vs Réalité pour {name} avec le modèle: {label}")
    plt.xlabel("Jours")
    plt.ylabel("Cours (déscalé)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join("Projet Final","Regression", "ML","Visualisation", f"pred_vs_real_for{name}_{label}.png"))
    plt.close()


def evaluate_all_models(models, X_test: np.ndarray, y_test: np.ndarray, scaler: MinMaxScaler, name):
    best_model = None
    best_rmse_score = np.inf
    best_model_name = ""
    result = {}
    for model_name, model in models.items():
        mse, rmse, preds, y_test_inv = evaluate_model(model, X_test, y_test, scaler)
        result[model_name] = {"MSE":mse, "RMSE":rmse}

        if rmse < best_rmse_score:
            best_rmse_score = rmse
            best_model = model
            best_model_name = model_name

    plot_predictions(y_test_inv, len(y_test), preds,best_model_name,name)
    return best_model_name, best_model, best_rmse_score, result

def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return obj

def give_next_close_all_companies():
    dico = data_for_all_company("Projet Final\\Companies_historical_data")
    dico_next_close = {}
    dico_evals_models = {}
    for name, value in dico.items():
        [X_train, X_test, y_train, y_test, scaler, X_last_day] = value
        models = get_best_models(X_train, y_train)
        best_model_name, best_model, best_rmse_score, eval_models = evaluate_all_models(models, X_test, y_test, scaler, name)

        dico_evals_models[name] = {"best model":best_model_name, "metrique evaluation":eval_models}

        pred = best_model.predict(X_last_day)
        dico_next_close[name] = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()[0]

    save_path = os.path.join("Projet Final", "Regression", "ML","Resultats","next_close_companies.json")
    save_path_2 = os.path.join("Projet Final", "Regression", "ML","Evaluation","best_model_companies.json")

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(dico_next_close, f, ensure_ascii=False, indent=4,default=convert_numpy)

    with open(save_path_2, 'w', encoding='utf-8') as f:
        json.dump(dico_evals_models, f, ensure_ascii=False, indent=4,default=convert_numpy)

    return dico_next_close