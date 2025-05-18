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
import json
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


def elbow_method_kmeans(data: pd.DataFrame, filename: str, k_range: range = range(3, 25)) -> None:
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
    plt.xticks(range(3, 25))
    plt.grid(True)
    plt.savefig(os.path.join("Projet Final", "Clustering", "Fine-tuning", f"Elbow_Kmeans_{filename}.png"))
    plt.close() 

    return inertias

def silhouette_method_kmeans(data: pd.DataFrame, filename: str, k_range: range = range(3, 25)) -> None:
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
    plt.xticks(range(3, 25))
    plt.grid(True)
    plt.savefig(os.path.join("Projet Final", "Clustering", "Fine-tuning", f"Silhouette_Kmeans_{filename}.png"))
    plt.close()

    return silhouette_scores

def calinski_harabasz_method_kmeans(data: pd.DataFrame, filename: str, k_range: range = range(3, 25),) -> None:
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
    plt.xticks(range(3, 25))
    plt.grid(True)
    plt.savefig(os.path.join("Projet Final", "Clustering", "Fine-tuning", f"Calinski_harabasz_Kmeans_{filename}.png"))
    plt.close()

    return calinski_scores


def silhouette_hac(data: pd.DataFrame, filename: str, k_range: range = range(3, 25)) -> list:
    silhouette_scores = []
    for k in k_range:
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)

    plt.plot(k_range, silhouette_scores, marker='o')
    plt.title("Silhouette Score - HAC")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.savefig(os.path.join("Projet Final", "Clustering", "Fine-tuning", f"Silhouette_HAC_{filename}.png"))
    plt.close()

    return silhouette_scores

def calinski_harabasz_hac(data: pd.DataFrame, filename: str, k_range: range = range(2, 25)) -> list:
    """
    Applique la méthode de Calinski-Harabasz pour HAC.

    Args:
        data (pd.DataFrame): Données standardisées.
        filename (str): Nom pour enregistrer l'image.
        k_range (range): Plage de k à tester.

    Returns:
        list: Liste des scores Calinski-Harabasz.
    """
    scores = []
    for k in k_range:
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(data)
        score = calinski_harabasz_score(data, labels)
        scores.append(score)

    # Plot
    plt.plot(k_range, scores, marker='o')
    plt.title("Score Calinski-Harabasz - HAC")
    plt.xlabel("Nombre de clusters")
    plt.ylabel("Score Calinski-Harabasz")
    plt.grid(True)
    plt.savefig(os.path.join("Projet Final", "Clustering", "Fine-tuning", f"Calinski_harabasz_HAC_{filename}.png"))
    plt.close()

    return scores

def get_optimal_cluster_kmean(data: pd.DataFrame, filename:str, k_range: range = range(3, 25), max_k_allowed: int = 15) -> int:

    inertias = elbow_method_kmeans(data, filename, k_range)
    sil_scores = silhouette_method_kmeans(data, filename, k_range)
    calinski = calinski_harabasz_method_kmeans(data, filename, k_range)
    best_k = k_range[np.argmax(sil_scores)]
    return best_k


def get_optimal_cluster_hac(data: pd.DataFrame, filename:str, k_range: range = range(3, 25), max_k_allowed: int = 15) -> int:
    sil_scores = silhouette_hac(data, filename, k_range)
    calinski = calinski_harabasz_hac(data, filename, k_range)
    best_k = k_range[np.argmax(sil_scores)]
    return best_k


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


def plot_tsne(data: pd.DataFrame, cluster_col: str, filename: str) -> None:
    """
    Représentation TSNE des clusters.

    Args:
        data (pd.DataFrame): Données avec colonne de cluster.
        cluster_col (str): Nom de la colonne indiquant les clusters.
    """
    tsne = TSNE(n_components=2, perplexity=18, random_state=42)
    reduced = tsne.fit_transform(data.drop(columns=[cluster_col]))
    plt.figure()
    plt.scatter(reduced[:, 0], reduced[:, 1], c=data[cluster_col], cmap='viridis')
    plt.title(f'Visualisation TSNE des clusters - {filename}')
    plt.savefig(os.path.join("Projet Final", "Clustering", "Visualisation", f"TSNE_{filename}.png"))
    plt.close()

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

def plot_dendrogram(data: pd.DataFrame, dataset_name: str, method: str = 'ward') -> None:
    linked = linkage(data, method=method)
    plt.figure(figsize=(10, 6))
    dendrogram(linked, labels=data.index.tolist(), leaf_rotation=90)
    plt.title(f'Dendrogramme des entreprises - {dataset_name}')
    plt.xlabel('Entreprise')
    plt.ylabel('Distance')
    plt.savefig(os.path.join("Projet Final", "Clustering", "Visualisation", f"dendrogram_{dataset_name}.png"))
    plt.close()



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
    plt.savefig(os.path.join("Projet Final","Clustering", "Visualisation", "dendogram_correlation_clustering.png"))
    plt.close()

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


def compare_clustering_algorithms_with_best_clusters(
    data: List[pd.DataFrame]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare KMeans, HAC et DBSCAN sur plusieurs datasets via le score de silhouette,
    et retourne aussi les clusters finaux du meilleur algo pour chaque dataset.

    Args:
        data (List[pd.DataFrame]): Liste de DataFrames à comparer.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - DataFrame des scores de silhouette.
            - DataFrame des clusters du meilleur algo pour chaque dataset.
    """
    

    dataset_names = ["ratios financiers", "rendements", "correlation des rendements"]

    results = []
    best_clusters = {}

    for i, dataset in enumerate(data):
        row = {'Dataset': dataset_names[i]}
        best_score = -1
        best_algo = None
        best_labels = None

        nb_cluster_kmeans= get_optimal_cluster_kmean(dataset,dataset_names[i])
        nb_cluster_hac = get_optimal_cluster_hac(dataset,dataset_names[i])
        #params_dbscan = find_best_dbscan_params(dataset)

        algorithms = {
        "KMeans": lambda: KMeans(n_clusters=nb_cluster_kmeans, random_state=42),
        "HAC": lambda: AgglomerativeClustering(n_clusters=nb_cluster_hac),
        "DBSCAN": lambda: DBSCAN(eps=1, min_samples=2)
        }

        for algo_name, algo_fn in algorithms.items():
            algo = algo_fn()
            try:
                clusters = algo.fit_predict(dataset)
                if algo_name == "DBSCAN" and len(set(clusters)) <= 1:
                    score = -1
                else:
                    score = silhouette_score(dataset, clusters)

            except:
                score = -1
                clusters = None

            row[f"{algo_name} - Silhouette"] = score

            if clusters is not None:
                df_clusters = dataset.copy()
                df_clusters["cluster"] = clusters

                if algo_name in ["KMeans", "DBSCAN"]:
                    plot_tsne(df_clusters, "cluster", f"{algo_name}_{dataset_names[i]}")
                elif algo_name == "HAC":
                    plot_dendrogram(dataset, dataset_names[i])
                    
            if score > best_score and clusters is not None:
                best_score = score
                best_algo = algo_name
                best_labels = clusters

        # Ajout des clusters du meilleur algo dans un DataFrame
        best_clusters[dataset_names[i]] = pd.Series(best_labels, index=dataset.index, name=f"Cluster ({best_algo})")

        results.append(row)

    results_df = pd.DataFrame(results)
    cluster_df = pd.concat(best_clusters.values(), axis=1)
    cluster_df.columns = dataset_names
    return results_df, cluster_df


def get_similar_companies_by_criteria(cluster_assignments: pd.DataFrame) -> dict:
    """
    Pour chaque entreprise, retourne les entreprises similaires selon les clusters
    obtenus par critère (ratios, rendements, corrélations).

    Args:
        cluster_assignments (pd.DataFrame): DataFrame où chaque colonne correspond au
                                            cluster d'un critère et les lignes sont les entreprises.

    Returns:
        dict: Dictionnaire {entreprise: {critère: [entreprises similaires]}}
    """
    similar_companies = {}

    for company in cluster_assignments.index:
        similar_companies[company] = {}
        for criterion in cluster_assignments.columns:
            cluster_label = cluster_assignments.loc[company, criterion]
            # Entreprises dans le même cluster, sauf elle-même
            same_cluster = cluster_assignments[cluster_assignments[criterion] == cluster_label].index
            similar_list = [c for c in same_cluster if c != company]
            similar_companies[company][criterion] = similar_list

    return similar_companies


def get_similar_companie_dico():
    # ratios financiers
    data = load_data("Projet Final\\companies_ratio.csv")

    # rendements journaliers
    returns = build_returns_dataframe("Projet Final\\Companies_historical_data")
    returns_t = returns.T

    # matrice de corrélation des rendements journaliers
    corr_returns = returns.corr()

    # feature selection et preprocessing
    colonne_to_keep = feature_selection(data)
    data_clean = preprocess_financial_data(data,colonne_to_keep)

     # comparaison des algorithmes
    results_df, cluster_df = compare_clustering_algorithms_with_best_clusters([data_clean,returns_t,corr_returns])
    cluster_df = cluster_df[~cluster_df.index.duplicated(keep='first')]
    similar_companies = get_similar_companies_by_criteria(cluster_df)

    # Chemin de sauvegarde
    save_path = os.path.join("Projet Final", "Clustering", "Resultats","similar_companies.json")
    csv_path = os.path.join("Projet Final", "Clustering", "Evaluation", "results_df.csv")

    # Sauvegarde 
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(similar_companies, f, ensure_ascii=False, indent=4)

    results_df.to_csv(csv_path, index=True)

    return similar_companies
