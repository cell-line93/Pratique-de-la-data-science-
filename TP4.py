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
    return [X_train, X_test, y_train, y_test, scaler]

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

def train_model(model, param_grid: Dict[str, List], X_train: np.ndarray, y_train: np.ndarray):
    """
        Effectue un GridSearch pour trouver les meilleurs hyperparamètres.
    """
    grid = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    return grid.best_estimator_

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

def plot_predictions(y_real_full: np.ndarray, y_train_len: int, preds: np.ndarray, label: str):
    """
        Affiche la courbe des vraies valeurs et des prédictions pour un modèle.
        Les prédictions sont décalées pour correspondre à leur position temporelle.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_real_full)), y_real_full, color='red', label='Valeurs réelles')
    plt.plot(range(y_train_len + 30, y_train_len + 30 + len(preds)),preds,label=label)
    plt.title("Prédictions vs Réalité")
    plt.xlabel("Jours")
    plt.ylabel("Cours (déscalé)")
    plt.legend()
    plt.grid(True)
    plt.show()

def compare_models_by_company(filepath: str) -> pd.DataFrame:
    """
        Compare les performances de plusieurs modèles de régression pour une entreprise.
        Affiche les résultats et retourne un DataFrame récapitulatif.
    """
    X_train, X_test, y_train, y_test, scaler = prepare_data(filepath)
    results = {}

    models = {
        'XGBoost': (
            XGBRegressor(verbosity=0),
            {'n_estimators': [50, 100], 'max_depth': [3, 5]}
        ),
        'RandomForest': (
            RandomForestRegressor(),
            {'n_estimators': [50, 100], 'max_depth': [3, 5]}
        ),
        'KNN': (
            KNeighborsRegressor(),
            {'n_neighbors': [3, 5, 7]}
        ),
        'LinearRegression': (
            LinearRegression(),
            {}
        )
    }

    for name, (model, params) in models.items():
        best_model = train_model(model, params, X_train, y_train)
        mse, rmse, preds, y_test_inv = evaluate_model(best_model, X_test, y_test, scaler)
        results[name] = {'MSE': mse, 'RMSE': rmse}
        plot_predictions(y_test_inv, len(y_train), preds, label=f"{name} Predictions")

    results_df = pd.DataFrame(results).T
    print(results_df)
    return results_df


def compare_models_all_company(path: str) -> Dict[str, pd.DataFrame]:
    """
    Applique la comparaison de modèles à toutes les entreprises du dossier.
    """
    filepaths = glob.glob(f"{path}/*.csv")
    print(f"{len(filepaths)} fichiers trouvés dans {path}")
    performance_dict: Dict[str, pd.DataFrame] = {}

    for file in filepaths:
        filename = os.path.basename(file)
        company_name = os.path.splitext(filename)[0].split('_')[0]
        performance_dict[company_name] = compare_models_by_company(file)

    return performance_dict


