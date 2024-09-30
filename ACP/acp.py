import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

class ACP:
    def __init__(self, df, target_column=None,point_size=4):
        self.df = df
        self.target_column = target_column
        self.Z = None
        self.acp = None
        self.coord1 = None
        self.varexpl = None
        self.modalites = None
        self.Quali = None
        self.p = None
        self.n = None
        self.colors = ['lightblue', 'tomato', 'yellow', 'green', 'gray', 'pink', 'brown', 'black']
        self.point_size = point_size
        self._prepare_data()
        self._fit_acp()

    def _prepare_data(self):
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

    def _fit_acp(self):
        self.acp = PCA(svd_solver='full')
        self.coord1 = self.acp.fit_transform(self.Z)
        self.varexpl = self.acp.explained_variance_ratio_ * 100
        self.eigval = (self.n - 1) / self.n * self.acp.explained_variance_
        sqrt_eigval = np.sqrt(self.eigval)
        self.corvar = np.zeros((self.p, self.p))
        for k in range(self.p):
            self.corvar[:, k] = self.acp.components_[k, :] * sqrt_eigval[k]

    @property
    def variance(self):
        return self.varexpl

    def valeur_propre(self):
        plt.figure(figsize=(5, 4))
        plt.bar(np.arange(len(self.varexpl)) + 1, self.varexpl)
        plt.plot(np.arange(len(self.varexpl)) + 1, self.varexpl.cumsum(), c="red", marker='o')
        plt.xlabel("# Axe Factorielle")
        plt.ylabel("Pourcentage d'inertie")
        plt.title("Eboulis des valeurs propres", fontsize=16)
        plt.show()

    def cercle_1(self):
        self._plot_cercle(0, 1, "Cercle des corrélations (1er plan factoriel)")

    def cercle_2(self):
        self._plot_cercle(2, 3, "Cercle des corrélations (2e plan factoriel)")

    def _plot_cercle(self, axe1, axe2, title):
        fig, axes = plt.subplots(figsize=(7, 7))
        axes.set_xlim(-1, 1)
        axes.set_ylim(-1, 1)
        for j in range(self.p):
            plt.arrow(0, 0, self.corvar[j, axe1], self.corvar[j, axe2], alpha=0.5, head_width=0.05, head_length=0.05)
            plt.text(self.corvar[j, axe1] * 1.15, self.corvar[j, axe2] * 1.15, self.Z.columns[j], color='black', ha='center', va='center')
        cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
        axes.add_artist(cercle)
        plt.xlabel(f"Axe {axe1+1} ({self.varexpl[axe1]:.1f}%)")
        plt.ylabel(f"Axe {axe2+1} ({self.varexpl[axe2]:.1f}%)")
        plt.title(title)
        plt.show()

    def individu(self):
        fig, axes = plt.subplots(figsize=(8,8))
        axes.set_xlim(-5,7)
        axes.set_ylim(-5,5)
        if self.target_column:
            for mod, color in zip(self.modalites, self.colors):
                plt.scatter(self.coord1[self.Quali == mod, 0], self.coord1[self.Quali == mod, 1], label=mod, color=color, alpha=0.4,marker="+",s=self.point_size)
        else:
            plt.scatter(self.coord1[:, 0], self.coord1[:, 1],s=self.point_size, alpha=0.4,marker=".")
        plt.xlabel("Axe 1")
        plt.ylabel("Axe 2")
        plt.plot([-6,7],[0,0],color='silver',linestyle='-',linewidth=1)
        plt.plot([0,0],[-6,6],color='silver',linestyle='-',linewidth=1)
        plt.title("Nuage des individus")
        plt.legend()
        plt.show()

