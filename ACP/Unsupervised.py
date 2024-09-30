import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

class Unsupervised:
    def __init__(self, df, clustering_method='kmeans', dimensionality_method='acp', use_sample=True, n_samples=2000,
                 kmeans_params=None, dbscan_params=None, tsne_params=None, acpnonlineaire_params=None):
        """
        Classe pour clustering et réduction de dimension avec ACP, ACP Non Linéaire, ou t-SNE.

        :param df: DataFrame à utiliser pour l'analyse.
        :param clustering_method: Méthode de clustering ('kmeans' ou 'dbscan').
        :param dimensionality_method: Méthode de réduction de dimension ('acp', 'acpnonlineaire', 'tsne').
        :param use_sample: Si True, utilise un échantillon du DataFrame pour l'analyse.
        :param n_samples: Nombre d'échantillons à prendre si use_sample est True.
        :param kmeans_params: Dictionnaire de paramètres pour KMeans.
        :param dbscan_params: Dictionnaire de paramètres pour DBSCAN.
        :param tsne_params: Dictionnaire de paramètres pour t-SNE.
        :param acpnonlineaire_params: Dictionnaire de paramètres pour ACP non linéaire (Kernel PCA).
        """
        self.df = df
        self.clustering_method = clustering_method
        self.dimensionality_method = dimensionality_method
        self.use_sample = use_sample
        self.n_samples = n_samples
        self.sample_df = None
        self.cluster_model = None
        self.reduction_model = None
        self.kmeans_params = kmeans_params or {}
        self.dbscan_params = dbscan_params or {}
        self.tsne_params = tsne_params or {}
        self.acpnonlineaire_params = acpnonlineaire_params or {}
        self._prepare_data()  # Prépare les données en les standardisant (centrage et réduction)
        self._fit_clustering()  # Effectue le clustering
        self._fit_dimensionality_reduction()  # Applique la méthode de réduction de dimension

    def _prepare_data(self):
        """
        Préparation des données : standardisation des données (centrage et réduction).
        Si use_sample est activé, prend un échantillon des données.
        """
        if self.use_sample and len(self.df) > self.n_samples:
            self.sample_df = self.df.sample(self.n_samples, random_state=42)
        else:
            self.sample_df = self.df

        # Centrage et réduction
        self.scaler = StandardScaler()
        self.sample_df[self.sample_df.columns] = self.scaler.fit_transform(self.sample_df[self.sample_df.columns])
        print("Données standardisées avec centering et scaling.")

    def _fit_clustering(self):
        """
        Applique le clustering en fonction de la méthode choisie (KMeans ou DBSCAN).
        Les paramètres peuvent être personnalisés via kmeans_params ou dbscan_params.
        """
        if self.clustering_method == 'kmeans':
            self.cluster_model = KMeans(**self.kmeans_params)
        elif self.clustering_method == 'dbscan':
            self.cluster_model = DBSCAN(**self.dbscan_params)
        else:
            raise ValueError(f"Méthode de clustering '{self.clustering_method}' non reconnue.")
        
        self.sample_df['cluster'] = self.cluster_model.fit_predict(self.sample_df)

    def _fit_dimensionality_reduction(self):
        """
        Applique la méthode de réduction de dimension choisie (ACP, ACPNonLineaire, ou t-SNE).
        Les classes ACP, ACPNonLineaire, et tSNE sont externes à ce script.
        """
        if self.dimensionality_method == 'acp':
            from acp import ACP  # Import de la classe ACP à partir du script séparé
            self.reduction_model = ACP(self.sample_df, target_column='cluster')
        elif self.dimensionality_method == 'acpnonlineaire':
            from acp_non_lineaire import ACPNonLineaire  # Import de la classe ACPNonLineaire
            self.reduction_model = ACPNonLineaire(self.sample_df, target_column='cluster', **self.acpnonlineaire_params)
        elif self.dimensionality_method == 'tsne':
            from tsne import tSNE  # Import de la classe tSNE
            self.reduction_model = tSNE(self.sample_df, target_column='cluster', **self.tsne_params)
        else:
            raise ValueError(f"Méthode de réduction de dimension '{self.dimensionality_method}' non reconnue.")

    def plot_individus(self):
        """
        Visualise les individus après réduction de dimension avec les clusters.
        """
        self.reduction_model.plot_individus()

    def calculate_elbow(self):
        """
        Méthode Elbow pour évaluer le nombre optimal de clusters (uniquement pour KMeans).
        """
        if self.clustering_method == 'kmeans':
            distortions = []
            for k in range(1, 11):
                km = KMeans(n_clusters=k)
                km.fit(self.sample_df.drop(columns='cluster'))
                distortions.append(km.inertia_)
            plt.plot(range(1, 11), distortions, marker='o')
            plt.xlabel('Nombre de clusters')
            plt.ylabel('Distortion')
            plt.title('Méthode Elbow')
            plt.show()
        else:
            raise ValueError("La méthode Elbow est uniquement applicable avec KMeans.")

    def calculate_silhouette(self):
        """
        Calcule et affiche le score de silhouette pour évaluer la qualité du clustering.
        """
        score = silhouette_score(self.sample_df.drop(columns='cluster'), self.sample_df['cluster'])
        print(f'Silhouette Score: {score:.2f}')
