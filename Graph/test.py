## Script à utiliser pour la création de graphique en POO par chatgpt

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Création d'un camembert d'une variable d'un df
# cette exmple fonctionne avec une variable qui à 2 classes
# il faut un code qui generalise
# avec la liste de couleur
# colors = ['blue', 'tomato', 'yellow', 'green', 'gray', 'pink', 'brown', 'black']
# L'objet doit me permettre de selectionner un titre et le figsize

labels =df['smoking'].value_counts(sort = True).index
sizes = df['smoking'].value_counts(sort = True)

colors = ["lightblue","tomato"]
explode = (0.1,0.07)

plt.figure(figsize=(6,6))
plt.pie(sizes, explode=explode, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90,)
plt.legend(["0 - Non fumeur", "1 - Fumeur"], loc="lower right",bbox_to_anchor=(1.7, 0.8))

plt.title('Part des fumeurs',fontsize=14)
plt.show()


# Histogramme simple
# L'objet doit me permettre de selectionner un titre et le figsize, x et y
fig, ax = plt.subplots(figsize=(8, 3))
ax.set_xlabel( "",size=0 )
sns.barplot(x = df_4.index , y = 'PNB/hab', data=df_4 , color = 'r')
plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
plt.title("PNB par habitant pour l'année 2015")


# Histogramme des variables numeriques 
# l'ogjet ne doit selectionner que les variables numerique (ex : int et float)
# L'objet doit me permettre de selectionner un titre et le figsize

plt.rc('font', size=10)
plt.rc('axes', labelsize=10, titlesize=12)
plt.rc('legend', fontsize=10)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)


df.hist(bins=50, figsize=(14,12))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

plt.show()

# Histogramme stratifier selon une target, ici la variable smoking
# Séparer les variables explicatives et la variable cible
# vérifier le nombre de classe de la target pour ajuster les couleurs
# ncols doit toujours être = 4 et nrows doit s'ajuster selon le nombre de variable numerique du df il faut donc le calculer
# L'objet doit me permettre de selectionner un titre et le figsize

feat_cols = df.drop(columns=["smoking"])
target_col = df.smoking

# Créer la grille de sous-graphiques
fig, axs = plt.subplots(nrows=6, ncols=4, figsize=(16, 16))

# Itérer à travers toutes les variables
for i, feat_idx in enumerate(feat_cols):
# Calculer les coordonnées de la sous-figure
    row_idx = i // 4
    col_idx = i % 4

    # Faire un histogramme pour la variable courante
    sns.histplot(data=feat_cols, x=feat_cols.iloc[:,i], hue=target_col, color=("b","r"),ax=axs[row_idx][col_idx], multiple="dodge")

# Ajuster l'espace entre les sous-figures
plt.tight_layout()
plt.show()

# Paireplot
# il faut pouvoir ajouster le sample
# il doit etre calculer pour conserver 2000 lignes
# je dois pouvoir choisir le hue, qui dans le code POO s'appelle target, si pas selectionner ca veut dire qu'il n'y en  a pas 
df_pairplot=df.sample(frac=0.01)
sns.pairplot(df_pairplot, hue="smoking", corner=True, plot_kws={'alpha':0.3})



# Parallele coordinate
# il faut pouvoir ajouster le sample
# il doit etre calculer pour conserver 2000 lignes
# Le sample doit être calculer pour avoir 2000 lignes max
# Centrer et reduire les données
# je dois pouvoir choisir le hue, qui dans le code POO s'appelle target (ds cet exemple classe), si pas selectionner ca veut dire qu'il n'y en  a pas 
from yellowbrick.features import ParallelCoordinates
from yellowbrick.features import ParallelCoordinates
from yellowbrick.datasets import load_occupancy

# Specify the features of interest and the classes of the target
features = list(df.columns)
classes = list(df["k_means"].unique())

# Instantiate the visualizer
visualizer = ParallelCoordinates(
                classes=classes, features=features,
                sample=0.08, shuffle=True)
# normalize='standard'
# Fit the visualizer and display it
visualizer.fit_transform(df, df["k_means"])
visualizer.show()


# Visualisation de la corrélation entre les variables
df_corr = df.corr(method='pearson')
plt.figure(figsize=(12, 11))
sns.heatmap(df_corr,fmt=".2f", annot=True,cmap="vlag",center=0)
plt.show()


# Une courbe et sa moyenne mobile
# L'objet doit me permettre de selectionner un titre et le figsize, le resemple, ici w pour week, je dois pouvoir choisir entre day month week et year
# et la valeur de window

plt.figure(figsize=(14,8))
df["price"].resample("w").sum().plot(label="Price",ls='--',lw=1,color='b',alpha=0.8)
agg_total["sum"].rolling(window=8).mean().plot(label="Moyenne mobile",color='r',alpha=0.8)


plt.title("Chiffre d'affaires hebdomadaire et lissage par moyenne mobile")
plt.legend()
plt.show()


# Valeurs manquantes, graphe general
import missingno as msno
%matplotlib inline
msno.matrix(df,color=(0.4,0,1))

# Valeurs manquantes variable par variable
# Création d'une fonction qui permet la représentation graphique des valeurs nulles

def graph_null(df,long):
    
   
    val_nul = pd.DataFrame(round(df.isnull().sum()/df.shape[0],3),columns=['% val_nulle'])
    val_nul = val_nul.sort_values("% val_nulle", ascending=False)

    val_nul=val_nul*100
    val_nul

    fig, ax = plt.subplots(figsize=(6, long))
    
    ax.set_xlim(0, 100)
    ax.set_xlabel( "",size=0 )
    sns.barplot(x =val_nul['% val_nulle']  , y =val_nul.index ,data = val_nul , color = 'r',orient='h')
    plt.setp(ax.get_yticklabels(),fontsize = 8)
    plt.title("% de valeur manquante par colonne")


    