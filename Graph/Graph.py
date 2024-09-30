import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from yellowbrick.features import ParallelCoordinates

class Graphiques:
    def __init__(self, df):
        self.df = df

    def camembert(self, column, title='', figsize=(6, 6), colors=None):
        labels = self.df[column].value_counts(sort=True).index
        sizes = self.df[column].value_counts(sort=True)
        colors = colors if colors else ['lightblue', 'tomato', 'yellow', 'green', 'gray', 'pink', 'brown', 'black'][:len(labels)]
        explode = [0.1 if i == 0 else 0.07 for i in range(len(labels))]

        plt.figure(figsize=figsize)
        plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.legend(labels, loc="lower right", bbox_to_anchor=(1.7, 0.8))
        plt.title(title, fontsize=14)
        plt.show()

    def histogramme_simple(self, x, y, title='', figsize=(8, 3), color='r'):
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x=x, y=y, data=self.df, color=color, ax=ax)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        plt.title(title)
        plt.show()

    def histogramme_numerique(self, title='', figsize=(14, 12)):
        num_df = self.df.select_dtypes(include=[np.number])
        num_df.hist(bins=50, figsize=figsize)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        plt.suptitle(title)
        plt.show()

    def histogramme_stratifie(self, target, title='', figsize=(16, 16)):
        feat_cols = self.df.drop(columns=[target])
        target_col = self.df[target]
        num_cols = feat_cols.select_dtypes(include=[np.number]).columns
        nrows = (len(num_cols) + 3) // 4

        fig, axs = plt.subplots(nrows=nrows, ncols=4, figsize=figsize)
        for i, col in enumerate(num_cols):
            row_idx = i // 4
            col_idx = i % 4
            sns.histplot(data=self.df, x=col, hue=target_col, ax=axs[row_idx, col_idx], multiple="dodge")

        plt.tight_layout()
        plt.suptitle(title)
        plt.show()

    def paireplot(self, target=None, sample_frac=0.01):
        df_sample = self.df.sample(frac=sample_frac)
        sns.pairplot(df_sample, hue=target, corner=True, plot_kws={'alpha':0.3})
        plt.show()

    def parallele_coordinates(self, target, sample_frac=0.01):
        df_sample = self.df.sample(frac=sample_frac)
        features = list(df_sample.columns)
        classes = list(df_sample[target].unique())

        scaler = StandardScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df_sample), columns=features)

        visualizer = ParallelCoordinates(
            classes=classes, features=features, sample=1.0, shuffle=True
        )
        visualizer.fit(df_normalized, df_sample[target])
        visualizer.show()

    def heatmap_corr(self, method='pearson', figsize=(12, 11)):
        df_corr = self.df.corr(method=method)
        plt.figure(figsize=figsize)
        sns.heatmap(df_corr, fmt=".2f", annot=True, cmap="vlag", center=0)
        plt.title("Heatmap des corrélations")
        plt.show()

    def courbe_moyenne_mobile(self, column, window=8, resample='W', title='', figsize=(14, 8)):
        plt.figure(figsize=figsize)
        self.df[column].resample(resample).sum().plot(label=column, ls='--', lw=1, color='b', alpha=0.8)
        self.df[column].resample(resample).sum().rolling(window=window).mean().plot(label="Moyenne mobile", color='r', alpha=0.8)
        plt.title(title)
        plt.legend()
        plt.show()

    def valeurs_manquantes(self):
        import missingno as msno
        msno.matrix(self.df, color=(0.4, 0, 1))
        plt.show()

    def valeurs_manquantes_par_colonne(self, long=10):
        val_nul = pd.DataFrame(round(self.df.isnull().sum()/self.df.shape[0], 3), columns=['% val_nulle'])
        val_nul = val_nul.sort_values("% val_nulle", ascending=False) * 100
        fig, ax = plt.subplots(figsize=(6, long))
        ax.set_xlim(0, 100)
        sns.barplot(x=val_nul['% val_nulle'], y=val_nul.index, color='r', orient='h')
        plt.setp(ax.get_yticklabels(), fontsize=8)
        plt.title("% de valeur manquante par colonne")
        plt.show()

# Exemple d'utilisation :
# from Graph import Graphiques
# df = pd.read_csv('votre_fichier.csv')  # Charger les données
# graph = Graphiques(df)
# graph.camembert('smoking', title='Part des fumeurs')
# graph.histogramme_simple(x='colonne_x', y='colonne_y', title='Titre de l\'histogramme')
# graph.histogramme_numerique(title='Histogramme des variables numériques')
# graph.histogramme_stratifie(target='smoking', title='Histogramme stratifié par smoking')
# graph.paireplot(target='smoking')
# graph.parallele_coordinates(target='k_means')
# graph.heatmap_corr()
# graph.courbe_moyenne_mobile(column='price', title='Chiffre d\'affaires hebdomadaire')
# graph.valeurs_manquantes()
# graph.valeurs_manquantes_par_colonne()
