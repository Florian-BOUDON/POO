import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
from sklearn.metrics import trustworthiness
import numpy as np

class tSNE:
    def __init__(self, df, target_column=None, point_size=4, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, random_state=None):
        """
        Classe pour l'application du t-SNE (t-distributed Stochastic Neighbor Embedding).
        :param df: DataFrame contenant les données
        :param target_column: Colonne cible pour les données qualitatives (optionnelle)
        :param point_size: Taille des points pour la visualisation
        :param n_components: Nombre de composantes à utiliser dans la réduction (par défaut 2 pour la visualisation)
        :param perplexity: Perplexité utilisée par le t-SNE (influence le voisinage)
        :param learning_rate: Taux d'apprentissage pour optimiser le t-SNE
        :param n_iter: Nombre d'itérations pour optimiser le t-SNE
        :param random_state: Graine aléatoire pour reproductibilité
        """
        self.df = df
        self.target_column = target_column
        self.Z = None
        self.tsne = None
        self.coord1 = None
        self.modalites = None
        self.Quali = None
        self.p = None
        self.n = None
        self.colors = ['lightblue', 'tomato', 'yellow', 'green', 'gray', 'pink', 'brown', 'black']
        self.point_size = point_size
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state

        self._prepare_data()
        self._fit_tsne()

    def _prepare_data(self):
        """
        Préparation des données : centrage et réduction
        """
        sc = StandardScaler()
        if self.target_column:
            self.Z = sc.fit_transform(self.df.drop(columns=[self.target_column]))
            self.Z = pd.DataFrame(self.Z, columns=self.df.columns.drop(self.target_column))
            self.Quali = self.df[self.target_column]
            self.modalites = np.unique(self.df[self.target_column])
            nb_color = len(self.modalites)
            self.colors = self.colors[:nb_color]
        else:
            self.Z = sc.fit_transform(self.df)
            self.Z = pd.DataFrame(self.Z, columns=self.df.columns)
            self.Quali = None
            self.modalites = None
        self.p = self.Z.shape[1]
        self.n = self.Z.shape[0]

    def _fit_tsne(self):
        """
        Applique t-SNE aux données standardisées.
        """
        print("Utilisation du t-SNE avec perplexité:", self.perplexity)
        self.tsne = TSNE(n_components=self.n_components, perplexity=self.perplexity, 
                         learning_rate=self.learning_rate, n_iter=self.n_iter, random_state=self.random_state)
        self.coord1 = self.tsne.fit_transform(self.Z)

    def cercle_1(self):
        """
        Visualisation des données dans le premier plan t-SNE (axes 1 et 2)
        """
        self._plot_projection(0, 1, "Projection des données t-SNE (1er plan factoriel)")

    def cercle_2(self):
        """
        Visualisation des données dans le deuxième plan t-SNE (axes 2 et 3) si les composantes sont en 3D
        """
        if self.n_components > 2:
            self._plot_projection(1, 2, "Projection des données t-SNE (2e plan factoriel)")
        else:
            print("La projection t-SNE a seulement", self.n_components, "composantes. Impossible de visualiser plus de 2 dimensions.")

    def _plot_projection(self, axe1, axe2, title):
        """
        Fonction privée pour afficher la projection des données selon deux axes principaux du t-SNE.
        """
        fig, axes = plt.subplots(figsize=(8, 8))
        axes.set_xlim(-50, 50)
        axes.set_ylim(-50, 50)
        if self.target_column:
            for mod, color in zip(self.modalites, self.colors):
                plt.scatter(self.coord1[self.Quali == mod, axe1], self.coord1[self.Quali == mod, axe2],
                            label=mod, color=color, alpha=0.4, marker="+", s=self.point_size)
        else:
            plt.scatter(self.coord1[:, axe1], self.coord1[:, axe2], s=self.point_size, alpha=0.4, marker=".")
        plt.xlabel(f"Axe {axe1+1}")
        plt.ylabel(f"Axe {axe2+1}")
        plt.plot([-50, 50], [0, 0], color='silver', linestyle='-', linewidth=1)
        plt.plot([0, 0], [-50, 50], color='silver', linestyle='-', linewidth=1)
        plt.title(title)
        plt.legend()
        plt.show()

    def individu(self):
        """
        Affiche la projection des individus dans le plan t-SNE.
        """
        fig, axes = plt.subplots(figsize=(8, 8))
        axes.set_xlim(-50, 50)
        axes.set_ylim(-50, 50)
        if self.target_column:
            for mod, color in zip(self.modalites, self.colors):
                plt.scatter(self.coord1[self.Quali == mod, 0], self.coord1[self.Quali == mod, 1],
                            label=mod, color=color, alpha=0.4, marker="+", s=self.point_size)
        else:
            plt.scatter(self.coord1[:, 0], self.coord1[:, 1], s=self.point_size, alpha=0.4, marker=".")
        plt.xlabel("Axe 1")
        plt.ylabel("Axe 2")
        plt.plot([-50, 50], [0, 0], color='silver', linestyle='-', linewidth=1)
        plt.plot([0, 0], [-50, 50], color='silver', linestyle='-', linewidth=1)
        plt.title("Nuage des individus t-SNE")
        plt.legend()
        plt.show()

    def trustworthiness_score(self):
        """
        Calcule la fidélité des voisins (trustworthiness) pour évaluer la qualité de la réduction dimensionnelle.
        """
        trust = trustworthiness(self.Z, self.coord1, n_neighbors=5)
        print(f"Fidélité des voisins (trustworthiness) : {trust:.3f}")
        return trust
