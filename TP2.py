import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import glob
import os
from typing import Tuple, List, Dict
from sklearn.metrics import calinski_harabasz_score,silhouette_score


# ----------------------------------
# 1.1 Financial Profiles Clustering
# ----------------------------------

def load_data(path: str) -> pd.DataFrame :
    """
        Charge les données financières depuis un fichier CSV.

        Args:
            path (str): Chemin vers le fichier CSV.

        Returns:
            pd.DataFrame: Données chargées avec l'index correctement défini.
    """
    try:
        data = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Erreur lors du chargement du fichier : {e}")
    
    # Vérifie que la colonne 'Unnamed: 0' existe
    if "Unnamed: 0" not in data.columns:
        raise KeyError("'Unnamed: 0' doit être une colonne du fichier CSV")

    data.set_index("Unnamed: 0", inplace=True)

    # Vérifie que le DataFrame n'est pas vide
    if data.empty:
        raise ValueError("Le fichier CSV est vide ou ne contient pas de données valides.")

    # Vérifie qu'il y a au moins une colonne restante
    if data.shape[1] == 0:
        raise ValueError("Le fichier ne contient pas de colonnes exploitables après mise en index.")
    
    return data 


def feature_selection(df: pd.DataFrame) -> List:
    """
    Récuperer la liste de colonnes à considérer 

    Args:
        dataframe

    Returns:
        list: liste des colonnes sélectionnées.
    """
    colonnes = df.columns

    # Colonnes contenant au moins un élément non numérique
    colonnes_non_numeriques = [col for col in colonnes if not pd.api.types.is_numeric_dtype(df[col])]

    # Colonnes contenant un certain pourcentage de valeurs manquantes
    seuil = int(len(df)/4) # un quart des lignes du dataframe
    colonnes_nan = colonnes[df.isna().sum() >= seuil].tolist()

    # Colonnes corrélées 
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    colonne_corr = [column for column in upper_tri.columns if any(upper_tri[column] > 0.9)]

    colonne_to_drop = list(set(colonnes_non_numeriques)|set(colonnes_nan)|set(colonne_corr))
    colonnes_to_keep = colonnes.difference(colonne_to_drop)

    return list(colonnes_to_keep)

def preprocess_financial_data(df: pd.DataFrame, selected_columns: List[str]) -> pd.DataFrame:
    """
    Prétraitement des données financières : sélection de colonnes pertinentes,
    suppression des valeurs manquantes et standardisation.

    Args:
        df (pd.DataFrame): Données financières initiales.
        selected_columns (List[str]): Colonnes à utiliser pour le clustering.

    Returns:
        pd.DataFrame: Données standardisées et nettoyées.
    """

    # suppression des NaN
    data = df[selected_columns].dropna()
    assert(data.isnull().sum().sum()==0)

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return pd.DataFrame(scaled_data, index=data.index, columns=selected_columns)


def elbow_method(data: pd.DataFrame, k_range: range = range(2, 25)) -> None:
    """
    Méthode du coude pour déterminer le nombre optimal de clusters.

    Args:
        data (pd.DataFrame): Données standardisées.
        k_range (range): Plage de valeurs pour k.
    """
    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    plt.plot(k_range, inertias, marker='o')
    plt.title('Méthode du coude')
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Inertie')
    plt.xticks(range(2, 25))
    plt.grid(True)
    plt.show()  

def silhouette_method(data: pd.DataFrame, k_range: range = range(2, 25)) -> None:
    """
    Méthode de l'indice de silhouette pour déterminer le nombre optimal de clusters.

    Args:
        data (pd.DataFrame): Données standardisées.
        k_range (range): Plage de valeurs pour k.
    """
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        score = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(score)

    plt.plot(k_range, silhouette_scores, marker='o')
    plt.title("Méthode de l'indice de silhouette")
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Score de silhouette')
    plt.xticks(range(2, 25))
    plt.grid(True)
    plt.show()

def calinski_harabasz_method(data: pd.DataFrame, k_range: range = range(2, 25)) -> None:
    """
    Méthode du score de Calinski-Harabasz pour déterminer le nombre optimal de clusters.

    Args:
        data (pd.DataFrame): Données standardisées.
        k_range (range): Plage de valeurs pour k.
    """
    calinski_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        score = calinski_harabasz_score(data, kmeans.labels_)
        calinski_scores.append(score)

    plt.plot(k_range, calinski_scores, marker='o')
    plt.title("Méthode du score de Calinski-Harabasz")
    plt.xlabel('Nombre de clusters')
    plt.ylabel('Score de Calinski-Harabasz')
    plt.xticks(range(2, 25))
    plt.grid(True)
    plt.show()

def do_kmeans_clustering(data: pd.DataFrame, n_clusters: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Applique KMeans et retourne les données avec cluster assigné.

    Args:
        data (pd.DataFrame): Données standardisées.
        n_clusters (int): Nombre de clusters à créer.

    Returns:
        Tuple[pd.DataFrame, KMeans]: Données avec clusters + modèle KMeans.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    numerical_columns = data.select_dtypes(include="number").columns
    data['cluster'] = kmeans.fit_predict(data)
    cluster_centers = pd.DataFrame(kmeans.cluster_centers_,columns=numerical_columns)

    return data, cluster_centers


def plot_tsne(data: pd.DataFrame, cluster_col: str = 'cluster') -> None:
    """
    Représentation TSNE des clusters.

    Args:
        data (pd.DataFrame): Données avec colonne de cluster.
        cluster_col (str): Nom de la colonne indiquant les clusters.
    """
    tsne = TSNE(n_components=2, perplexity=24, random_state=42)
    reduced = tsne.fit_transform(data.drop(columns=[cluster_col]))
    plt.scatter(reduced[:, 0], reduced[:, 1], c=data[cluster_col], cmap='viridis')
    plt.title('Visualisation TSNE des clusters')
    plt.show()


# -------------------------------
# 1.2 Risk Profiles Clustering
# -------------------------------

def do_hierarchical_clustering(data: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """
    Applique le clustering hiérarchique sur les données.

    Args:
        data (pd.DataFrame): Données standardisées.
        n_clusters (int): Nombre de clusters souhaité.

    Returns:
        pd.DataFrame: Données avec clusters ajoutés.
    """
    model = AgglomerativeClustering(n_clusters=n_clusters)
    data['cluster'] = model.fit_predict(data)

    return data

def plot_dendrogram(data: pd.DataFrame, method: str = 'ward') -> None:
    """
    Affiche un dendrogramme basé sur linkage.

    Args:
        data (pd.DataFrame): Données standardisées.
        method (str): Méthode de linkage.
    """
    linked = linkage(data, method=method)
    dendrogram(linked, labels=data.index.tolist(), leaf_rotation=90)
    plt.title('Dendrogramme des entreprises')
    plt.xlabel('Entreprise')
    plt.ylabel('Distance')
    plt.show()


# -------------------------------------------
# 1.3 Daily Returns Correlations Clustering
# -------------------------------------------

def build_returns_dataframe(path: str) -> pd.DataFrame:
    """
    Construit un DataFrame à partir des rendements journaliers de fichiers CSV.

    Args:
        path (str): Chemin vers le dossier contenant les fichiers CSV.

    Returns:
        pd.DataFrame: DataFrame des rendements journaliers.
    """
    filepaths = glob.glob(f"{path}/*.csv")
    print(f"{len(filepaths)} fichiers trouvés dans {path}")
    returns_dict: Dict[str, pd.Series] = {}

    for file in filepaths:
        df = pd.read_csv(file)
        filename = os.path.basename(file)
        filename = os.path.splitext(filename)[0].split('_')[0]
        returns_dict[filename] = df['Rendement']

    returns_df = pd.DataFrame(returns_dict)
    returns_df = returns_df.fillna(returns_df.mean())
    return returns_df


def correlation_clustering(returns_df: pd.DataFrame) -> None:
    """
    Applique un clustering hiérarchique sur la matrice de corrélation des rendements.

    Args:
        returns_df (pd.DataFrame): DataFrame des rendements journaliers.
    """
    corr_matrix = returns_df.corr()
    distance_matrix = 1 - corr_matrix #plus les rendements sont corrélés plus la distance est proche
    linked = linkage(squareform(distance_matrix), method='ward')
    dendrogram(linked, labels=corr_matrix.columns, leaf_rotation=90)
    plt.title('Clustering basé sur la corrélation des rendements')
    plt.xlabel('Entreprise')
    plt.ylabel('Distance')
    plt.show()

def do_dbscan_clustering(data: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> pd.DataFrame:
    """
    Applique DBSCAN sur les données.

    Args:
        data (pd.DataFrame): Données standardisées.
        eps (float): Distance maximale pour être considéré comme voisin.
        min_samples (int): Nombre minimum de points pour former un cluster.

    Returns:
        pd.DataFrame: Données avec colonne 'cluster'.
    """
    db = DBSCAN(eps=eps, min_samples=min_samples)
    data['cluster'] = db.fit_predict(data)
    return data

def find_best_dbscan_params(data: pd.DataFrame, eps_range: np.ndarray = np.arange(0.01, 2.1, 0.01), min_samples_range: range = range(2, 15)) -> dict:
    """
    Trouve les meilleurs paramètres pour DBSCAN en utilisant la méthode silhouette pour évaluer la qualité du clustering.

    Args:
        data (pd.DataFrame): Données standardisées.
        eps_range (np.ndarray): Plage de valeurs pour eps.
        min_samples_range (range): Plage de valeurs pour min_samples.

    Returns:
        dict: Meilleurs paramètres `eps` et `min_samples` ainsi que le score de silhouette correspondant.
    """
    best_score = -1
    best_eps = None
    best_min_samples = None
    
    # Itération sur les différentes valeurs de eps et min_samples
    for eps in eps_range:
        for min_samples in min_samples_range:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = db.fit_predict(data)

            # Calcul du score de silhouette (c'est un indicateur de la qualité du clustering)
            # Note : Si tous les points sont dans un seul cluster, silhouette_score échouera. On vérifie cela.
            if len(set(clusters)) > 1:  # Il faut au moins 2 clusters pour calculer le score
                try:
                    score = silhouette_score(data, clusters)
                    if score > best_score:
                        best_score = score
                        best_eps = eps
                        best_min_samples = min_samples
                except:
                    continue  # Si le calcul échoue, on passe à l'itération suivante.

    return {'best_eps': best_eps, 'best_min_samples': best_min_samples, 'best_score': best_score}


def compare_clustering_algorithms(data: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Compare KMeans, HAC (Agglomerative Clustering) et DBSCAN sur plusieurs datasets,
    en affichant uniquement les scores de silhouette.

    Args:
        data (List[pd.DataFrame]): Liste de DataFrames à comparer.

    Returns:
        pd.DataFrame: Tableau comparatif des scores de silhouette des algorithmes sur les datasets.
    """
    # Initialisation des algorithmes
    # Paramètres des modèles entrés manuellement
    algorithms = {
        "KMeans": KMeans(n_clusters=9, random_state=42),
        "HAC": AgglomerativeClustering(n_clusters=9),
        "DBSCAN": DBSCAN(eps=1.93, min_samples=2) 
    }
    Dataset_name = ["ratios financiers", "rendements", "correlation des rendements"]
    # Liste pour stocker les résultats
    results = []

    # Application des algorithmes sur chaque dataset
    for i, dataset in enumerate(data):
        row = {'Dataset': f"{Dataset_name[i]}"}

        for algo_name, algo in algorithms.items():
            if algo_name == "DBSCAN":
                # DBSCAN peut avoir des résultats avec -1 pour les points bruyants
                clusters = algo.fit_predict(dataset)
                # Si DBSCAN trouve des points bruyants (-1), on ne calcule pas le score silhouette
                if len(set(clusters)) > 1:
                    score = silhouette_score(dataset, clusters)
                else:
                    score = -1  # Score invalide pour DBSCAN si tout est un seul cluster ou bruit
            else:
                clusters = algo.fit_predict(dataset)
                score = silhouette_score(dataset, clusters)

            row[f"{algo_name} - Silhouette"] = score

        results.append(row)

    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results)
    return results_df


if __name__ == "__main__":

    # load data
        # ratios financiers
    data = load_data(r"C:\Users\tagob\Documents\DAUPHINE\Pratique de la data science\TP\TP1\companies_ratio.csv")
        # rendements journaliers
    returns = build_returns_dataframe(r"C:\Users\tagob\Documents\DAUPHINE\Pratique de la data science\TP\TP1\Companies_historical_data")
        # matrice de corrélation des rendements journaliers
    corr_returns = returns.corr()

    # feature selection et preprocessing
    colonne_to_keep = feature_selection(data)
    data_clean = preprocess_financial_data(data,colonne_to_keep)

    # comparaison des algorithmes
    print(compare_clustering_algorithms([data_clean,returns,corr_returns]))
    
