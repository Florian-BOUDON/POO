import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from sklearn.random_projection import GaussianRandomProjection
import numpy as np

class ACPNonLineaire:
    def __init__(self, df, target_column=None, point_size=4, kernel='rbf', gamma=None, degree=3, coef0=1, rdn=False, n_components=None):
        """
        Classe pour l'ACP non linéaire (Kernel PCA) ou projection aléatoire gaussienne.
        :param df: DataFrame contenant les données
        :param target_column: Colonne cible pour les données qualitatives (optionnelle)
        :param point_size: Taille des points pour la visualisation
        :param kernel: Type de noyau ('linear', 'poly', 'rbf', 'sigmoid', etc.)
        :param gamma: Coefficient pour le noyau RBF ou polynomial
        :param degree: Degré pour le noyau polynomial
        :param coef0: Coefficient pour le noyau polynomial ou sigmoid
        :param rdn: Si True, utilise GaussianRandomProjection au lieu de KernelPCA
        :param n_components: Nombre de composantes à utiliser (pour KernelPCA et GaussianRandomProjection)
        """
        self.df = df
        self.target_column = target_column
        self.Z = None
        self.kpca = None
        self.coord1 = None
        self.modalites = None
        self.Quali = None
        self.p = None
        self.n = None
        self.colors = ['lightblue', 'tomato', 'yellow', 'green', 'gray', 'pink', 'brown', 'black']
        self.point_size = point_size
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.rdn = rdn
        self.n_components = n_components

        self._prepare_data()
        self._fit_model()

    def _prepare_data(self):
        """
        Préparation des données, centrage et réduction
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

    def _fit_model(self):
        """
        Applique soit le Kernel PCA soit la projection aléatoire gaussienne en fonction de l'option choisie.
        """
        if self.rdn:
            # Utilisation de la projection aléatoire gaussienne
            print("Utilisation de la projection aléatoire gaussienne")
            self.kpca = GaussianRandomProjection(n_components=self.n_components if self.n_components else self.p)
            self.coord1 = self.kpca.fit_transform(self.Z)
        else:
            # Utilisation du Kernel PCA
            print("Utilisation de Kernel PCA avec le noyau:", self.kernel)
            self.kpca = KernelPCA(n_components=self.n_components if self.n_components else self.p, 
                                  kernel=self.kernel, gamma=self.gamma, degree=self.degree, coef0=self.coef0)
            self.coord1 = self.kpca.fit_transform(self.Z)

    def cercle_1(self):
        """
        Visualisation des données dans le premier plan factoriel (axes 1 et 2)
        """
        self._plot_projection(0, 1, "Projection des données (1er plan factoriel)")

    def cercle_2(self):
        """
        Visualisation des données dans le deuxième plan factoriel (axes 2 et 3)
        """
        self._plot_projection(2, 3, "Projection des données (2e plan factoriel)")

    def _plot_projection(self, axe1, axe2, title):
        """
        Fonction privée pour afficher la projection des données selon deux axes principaux.
        """
        fig, axes = plt.subplots(figsize=(8, 8))
        axes.set_xlim(-5, 5)
        axes.set_ylim(-5, 5)
        if self.target_column:
            for mod, color in zip(self.modalites, self.colors):
                plt.scatter(self.coord1[self.Quali == mod, axe1], self.coord1[self.Quali == mod, axe2],
                            label=mod, color=color, alpha=0.4, marker="+", s=self.point_size)
        else:
            plt.scatter(self.coord1[:, axe1], self.coord1[:, axe2], s=self.point_size, alpha=0.4, marker=".")
        plt.xlabel(f"Axe {axe1+1}")
        plt.ylabel(f"Axe {axe2+1}")
        plt.plot([-6, 7], [0, 0], color='silver', linestyle='-', linewidth=1)
        plt.plot([0, 0], [-6, 6], color='silver', linestyle='-', linewidth=1)
        plt.title(title)
        plt.legend()
        plt.show()

    def valeur_propre(self):
        """
        Visualisation de la variance expliquée par chaque composante.
        """
        if self.rdn:
            print("La méthode de projection aléatoire gaussienne ne permet pas de calculer la variance expliquée.")
        else:
            explained_variance = np.var(self.coord1, axis=0)
            explained_variance_ratio = explained_variance / np.sum(explained_variance) * 100
            
            plt.figure(figsize=(5, 4))
            plt.bar(np.arange(len(explained_variance_ratio)) + 1, explained_variance_ratio)
            plt.plot(np.arange(len(explained_variance_ratio)) + 1, np.cumsum(explained_variance_ratio), c="red", marker='o')
            plt.xlabel("# Axe Factorielle")
            plt.ylabel("Pourcentage d'inertie")
            plt.title("Éboulis des valeurs propres", fontsize=16)
            plt.show()

    def individu(self):
        """
        Affiche la projection des individus dans le plan factoriel.
        """
        fig, axes = plt.subplots(figsize=(8, 8))
        axes.set_xlim(-5, 5)
        axes.set_ylim(-5, 5)
        if self.target_column:
            for mod, color in zip(self.modalites, self.colors):
                plt.scatter(self.coord1[self.Quali == mod, 0], self.coord1[self.Quali == mod, 1],
                            label=mod, color=color, alpha=0.4, marker="+", s=self.point_size)
        else:
            plt.scatter(self.coord1[:, 0], self.coord1[:, 1], s=self.point_size, alpha=0.4, marker=".")
        plt.xlabel("Axe 1")
        plt.ylabel("Axe 2")
        plt.plot([-6, 7], [0, 0], color='silver', linestyle='-', linewidth=1)
        plt.plot([0, 0], [-6, 6], color='silver', linestyle='-', linewidth=1)
        plt.title("Nuage des individus")
        plt.legend()
        plt.show()

