import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Optional, List
import warnings
warnings.filterwarnings("ignore")
import os

# -----------------------------
# 1.1 Création du dataset
# -----------------------------

def create_target_features(data: np.ndarray, n_days: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(n_days, len(data)):
        x.append(data[i - n_days:i, 0])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

def prepare_data(filepath: str, n_days: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    df = pd.read_csv(filepath)
    df = df[["Close"]].dropna()
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    X, y = create_target_features(df_scaled, n_days=n_days)
    X_train, X_test = X[:int(len(X)*0.8)], X[int(len(X)*0.8):]
    y_train, y_test = y[:int(len(y)*0.8)], y[int(len(y)*0.8):]
    return X_train, X_test, y_train, y_test, scaler


# -----------------------------
# 1.2 Modèles de Deep Learning
# -----------------------------

def build_mlp_model(input_shape: Tuple[int], hidden_dims: int = 64, dropout_rate: float = 0.2, activation: str = 'relu', optimizer: str = 'adam', learning_rate: float = 0.001) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(hidden_dims, activation=activation),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.get(optimizer), loss='mean_squared_error')
    return model

def build_rnn_model(input_shape: Tuple[int], hidden_dims: int = 64, dropout_rate: float = 0.2, activation: str = 'tanh', optimizer: str = 'adam', learning_rate: float = 0.001) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.SimpleRNN(hidden_dims, activation=activation, input_shape=input_shape),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.get(optimizer), loss='mean_squared_error')
    return model

def prepare_sequence_data(X: np.ndarray, y: np.ndarray, time_steps: int = 30):
    """
    Prépare les données X et y pour un modèle LSTM avec fenêtrage automatique.
    
    Arguments :
    - X : ndarray de forme (n_samples, n_features), les données explicatives
    - y : ndarray de forme (n_samples,), la variable cible
    - time_steps : int, nombre de pas de temps à inclure dans chaque séquence
    
    Retourne :
    - X_seq : ndarray (n_sequences, time_steps, n_features)
    - y_seq : ndarray (n_sequences,)
    """
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i+time_steps])
        y_seq.append(y[i+time_steps])  # La valeur cible à prédire après la séquence

    return np.array(X_seq), np.array(y_seq)

def build_lstm_model(input_shape: Tuple[int], hidden_dims: int = 64, dropout_rate: float = 0.2, activation: str = 'tanh', optimizer: str = 'adam', learning_rate: float = 0.001) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(hidden_dims, activation=activation,input_shape=input_shape),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.get(optimizer), loss='mean_squared_error')
    return model

# -----------------------------
# 1.2.2 Entrainement des modèles
# -----------------------------

def train_model(model_type: str, X_train: np.ndarray, y_train: np.ndarray, input_shape: Tuple[int], hidden_dims: int = 64, dropout_rate: float = 0.2, activation: str = 'relu', optimizer: str = 'adam', epochs: int = 20, batch_size: int = 32) -> tf.keras.Model:
    if model_type == "MLP":
        model = build_mlp_model(input_shape, hidden_dims, dropout_rate, activation, optimizer)
    elif model_type == "RNN":
        model = build_rnn_model(input_shape, hidden_dims, dropout_rate, activation, optimizer)
    elif model_type == "LSTM":
        model = build_lstm_model(input_shape, hidden_dims, dropout_rate, activation, optimizer)
    else:
        raise ValueError("Model type must be one of: MLP, RNN, LSTM")

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return model

# -----------------------------
# 1.2.3 Prédiction
# -----------------------------

def predict(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray, scaler: MinMaxScaler, model_type: str):
    preds = model.predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
    preds_inv = scaler.inverse_transform(preds)

    mae = mean_absolute_error(y_test_inv, preds_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, preds_inv))

    print(f"Model: {model_type} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    print("Predictions vs Réelles:")
    for i in range(10):
        print(f"Prédit: {preds_inv[i][0]:.2f}, Réel: {y_test_inv[i][0]:.2f}")

    plt.plot(preds_inv, label='Prédictions')
    plt.plot(y_test_inv, label='Réelles')
    plt.legend()
    plt.title(f"{model_type} - Prédictions vs Réelles")
    plt.show()

    return mae, rmse


def compare_models(X_train, X_test, y_train, y_test, scaler, time_steps=20) -> pd.DataFrame:
    results = []

    # MLP
    model_mlp = train_model("MLP", X_train, y_train, input_shape=(X_train.shape[1],))
    mae_mlp, rmse_mlp = predict(model_mlp, X_test, y_test, scaler, "MLP")
    results.append({"Model": "MLP", "MAE": mae_mlp, "RMSE": rmse_mlp})

    # Séquences pour RNN et LSTM
    X_train_seq, y_train_seq = prepare_sequence_data(X_train, y_train, time_steps)
    X_test_seq, y_test_seq = prepare_sequence_data(X_test, y_test, time_steps)

    # RNN
    model_rnn = train_model("RNN", X_train_seq, y_train_seq, input_shape=(time_steps, X_train.shape[1]))
    mae_rnn, rmse_rnn = predict(model_rnn, X_test_seq, y_test_seq, scaler, "RNN")
    results.append({"Model": "RNN", "MAE": mae_rnn, "RMSE": rmse_rnn})

    # LSTM
    model_lstm = train_model("LSTM", X_train_seq, y_train_seq, input_shape=(time_steps, X_train.shape[1]))
    mae_lstm, rmse_lstm = predict(model_lstm, X_test_seq, y_test_seq, scaler, "LSTM")
    results.append({"Model": "LSTM", "MAE": mae_lstm, "RMSE": rmse_lstm})

    return pd.DataFrame(results)



if __name__ == "__main__":
    path = "C:\\Users\\tagob\\Documents\\DAUPHINE\\Pratique de la data science\\TP\\TP1\\Companies_historical_data"
    company = "Tesla"
    file_path = os.path.join(path, f"{company}_data.csv")
    X_train, X_test, y_train, y_test, scaler = prepare_data(file_path)
    result = compare_models(X_train,X_test,y_train,y_test,scaler)
    print(result)