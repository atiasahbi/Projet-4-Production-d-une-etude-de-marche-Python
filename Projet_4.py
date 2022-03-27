#!/usr/bin/env python
# coding: utf-8

# # Répertoire de travail:
# 

# In[1]:


cd "C:\Users\narje\Documents\Mon dossier Tableau\Sources de données"


# # Importation des librairies:

# In[2]:


import numpy as np #importation classique du numpy sous l'alias np
import pandas as pd #importation classique du pandas sous l'alias pd
from pandas import *
import  matplotlib.pyplot as plt #importation classique du module matplotlib.pyplot sous l'alias plt
from matplotlib.collections import LineCollection
from sklearn import *
from scipy.cluster.hierarchy import dendrogram
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import centroid,linkage, fcluster
from sklearn import preprocessing
from sklearn import cluster,metrics
from soccerplots.radar_chart import Radar
import seaborn as sns
from math import pi
from sklearn.cluster import DBSCAN
from collections import Counter
from sklearn.cluster import KMeans


# # Importer le fichier :

# In[3]:


DF = pd.read_csv('cleaned_DF_projet_9.csv',sep=';') #Importer le fichier cleaned_DF_projet_9 aprés nettoyage et préparartion sur Tableau Prep Builder
DF.describe(include='all')
DF.drop_duplicates(subset=['Zone'],inplace = True)
DF.shape
DF


# # Nettoyage du dataframe DF :

# In[4]:


plt.figure(figsize=(5,5)) #Carte chaleur des NaN
sns.heatmap(DF.isna(),cbar=False)
plt.show()
DF.shape


# In[5]:


DF.dropna(axis = 0 ,how = 'any' , inplace = True) # supprimer les lignes contenant des NaN
DF.shape


# In[6]:


plt.figure(figsize=(5,5)) #Carte chaleur des NaN
sns.heatmap(DF.isna(),cbar=False)
plt.show()


# In[7]:


(DF.isna().sum()/DF.shape[0]).sort_values() #Vérifier q'on a pas des valeurs manquantes


# In[8]:


# Analyses des outliers
for col in DF.select_dtypes('float'):
    plt.figure()
    DF.boxplot(column = [col])
    plt.show()


# In[9]:


# Courbe de distribution des variables quantitatives:
for col in DF.select_dtypes('float'):
    plt.figure()
    sns.distplot(DF[col])
    plt.show()


# # Fonctions :

# In[10]:


def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,7))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
        
def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(8,8))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_scree_plot(pca):
    scree = pca.explained_variance_ratio_*100
    plt.bar(np.arange(len(scree))+1, scree)
    plt.plot(np.arange(len(scree))+1, scree.cumsum(),c="red",marker='o')
    plt.xlabel("rang de l'axe d'inertie")
    plt.ylabel("pourcentage d'inertie")
    plt.title("Eboulis des valeurs propres")
    plt.show(block=False)

def plot_dendrogram(Z, names):
    plt.figure(figsize=(5,7))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('distance')
    dendrogram(
        Z,
        labels = names,
        orientation = "left",
    )
    plt.show()


# In[11]:


sns.pairplot(DF) # Matrice de correlation entre les # variables
plt.show()


# # Analyse en Composantes Principales (ACP):

# In[12]:


# Choix du nombre de composantes à calculer :

n_comp = 6

# selection des colonnes à prendre en compte dans l'ACP :

data_pca = DF[[
     'Importations - Volailles', 'Production-Volailles',
       'Disponibilité intérieure', 'Exportations - Volailles', 'PIB',
       'Population', 'Sous-alimentation (%)', 'Indice de stabilité',
       "Disponibilités protéines moyennes d'origine animale"
       ]]
data_pca
DF


# In[13]:



# Préparation des données pour l'ACP :

X = data_pca.values
names = DF.index
features = data_pca.columns

# Centrage et Réduction
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)

# Calcul des composantes principales
pca = PCA(n_components = n_comp)
pca.fit(X_scaled)


# In[14]:


# Vérifions la standardisation de données:
X_scaled = pd.DataFrame(data = X_scaled) #convertir X_scaled en dataframe
X_scaled.columns = data_pca.columns
for col in X_scaled.columns:
    plt.figure()
    sns.distplot(X_scaled[col])
    plt.show()


# In[15]:


print(np.std(X_scaled,axis=0)) # vérification moyennes nulles et les écarts-type unitaires

print(np.mean(X_scaled,axis=0))


# In[16]:


# Eboulis des valeurs propres

display_scree_plot(pca)

#Le porcentage (d'information) d'inertie expliqué par le 1er plan factoriel (1er axe + 2eme axe = 65%)


# In[17]:


# Cercle des corrélations
fig = plt.figure(figsize = (5,5))
pcs = pca.components_
display_circles(pcs, n_comp, pca, [(0,1),(2,3),(4,5)], labels = np.array(features))
plt.show()


# In[74]:


# Projection des individus
X_projected = pca.transform(X_scaled)
display_factorial_planes(X_projected, n_comp, pca, [(0,1),(1,2),(0,2)], alpha = 0.2,labels = np.array(names))
plt.show()
X_projected


# In[19]:


X_scaled


# # ACH

# In[20]:


# Clustering hiérarchique
Z = linkage(X_scaled, 'ward')


# In[21]:


# Affichage du dendrogramme
names = DF.index
plot_dendrogram(Z, names)
plt.show()


# In[22]:


# Analyse pour un nombre des clusters égale à 3 :

clusters3 = fcluster(Z, 3, criterion='maxclust') # Coupage du dendrogramme en 3 clusters
names = DF.Zone
C3 = pd.DataFrame({"Zone": names,"clusters": clusters3})

A = C3.groupby(['Zone']).sum()

print(C3)

# Effectifs par groupe :

names = DF.Zone
B = C3.groupby(['clusters']).count()
B = pd.DataFrame(data = B)
B.reset_index(inplace = True) #reindexation de dataframe B
B['clusters'] = B['clusters'].apply(str)
print(B,'\n')

P = pd.merge(C3, DF, how = 'inner', on = 'Zone') #jointure entre la table C et la table DF

N3 = P.groupby(['clusters']).mean() #Calcul de la moyenne de chaque variable pour chaque groupe

std_scale = preprocessing.StandardScaler().fit(N3)
N3_scaled = std_scale.transform(N3)
N3_scaled = pd.DataFrame(data = N3_scaled)

N3_scaled.columns = data_pca.columns
N3_scaled.index = np.arange(1, len(N3_scaled) + 1) #Commencer l'indexation de 1 au lieu de 0
N3_scaled['clusters'] = N3_scaled.index


N3_scaled.set_index('clusters',inplace=True)

N3_scaled


# In[23]:


# Heatmap avec les croisements entre les clusters de pays et les différentes variables:

N3_scaled.columns = data_pca.columns

plt.figure(figsize=(7,5))
sns.heatmap(N3_scaled, cmap = 'viridis') # Graphe carte chaleur des groupes des zones
plt.savefig("Carte chaleur avec les croisements entre les clusters de pays et les différentes variables ACH_3G.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[24]:


#Interprétations:

#print(C3[C3['group']==2]) #Filtre sur le groupe des pays à recommander
#Groupe 2 : pays qui importent les plus,qui ont un bon Indice de Stabilité et qui sont assez riche donc à recommander
F = C3[C3['clusters']==2]
F


# In[25]:


# Projection des individus sur le 1e plan factoriel ACH_3G:

plt.figure()

plt.scatter(X_projected[:,0], X_projected[:,1] , c=clusters3.astype(np.float))#Graphe nuage des points
plt.xlabel('F1')
plt.ylabel('F2')
plt.title("Projection des individus sur le 1e plan factoriel")

plt.show()


# In[26]:


# Analyse pour nombre des clusters égale à 5 :

clusters5 = fcluster(Z, 5, criterion = 'maxclust') # Coupage du dendrogramme en 5 clusters
names = DF.Zone
C5 = pd.DataFrame({"Zone": names,"Clusters": clusters5})
A = C5.groupby(['Zone']).sum()
print(C5)
# Effectifs par groupe :

names = DF.Zone
B = C5.groupby(['Clusters']).count()
B = pd.DataFrame(data = B)
B.reset_index(inplace = True) #reindexation de dataframe B
B['Clusters'] = B['Clusters'].apply(str)
print(B)

P = pd.merge(C5, DF, how = 'inner', on = 'Zone') #jointure entre la table C et la table DF

N5 = P.groupby(['Clusters']).mean() #Calcul de la moyenne de chaque variable pour chaque groupe

std_scale = preprocessing.StandardScaler().fit(N5)
N5_scaled = std_scale.transform(N5)
N5_scaled = pd.DataFrame(data = N5_scaled)

N5_scaled.columns = data_pca.columns
N5_scaled.index = np.arange(1, len(N5_scaled) + 1) # Commencer l'indexation de 1 au lieu de 0

N5_scaled['Clusters'] = N5_scaled.index
N5_scaled.set_index('Clusters',inplace=True)

N5_scaled


# In[27]:


# Heatmap avec les croisements entre les clusters de pays et les différentes variables:

N5_scaled.columns = data_pca.columns

plt.figure(figsize=(7,5))
sns.heatmap(N5_scaled, cmap = 'viridis') # Graphe carte chaleur des groupes des zones
plt.savefig("Carte chaleur avec les croisements entre les clusters de pays et les différentes variables ACH_5G.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[28]:


#Interprétations:

print(C5[C5['Clusters']==2]) #Filtre sur le groupe des pays à recommander
#Groupe 2 : pays qui importent les plus,qui ont un bon Indice de Stabilité et qui sont assez riche donc à recommander
C5[C5['Clusters']==2]


# In[29]:


# Projection des individus sur le 1e plan factoriel ACH_5G:

plt.figure()

plt.scatter(X_projected[:,0], X_projected[:,1],c=clusters5.astype(np.float))#Graphe nuage des points
plt.xlabel('F1')
plt.ylabel('F2')
plt.title("Projection des individus sur le 1e plan factoriel")

plt.show()


# In[30]:


# Analyse pour nombre des clusters égale à 7 :

clusters7 = fcluster(Z, 7, criterion='maxclust') # Coupage du dendrogramme en 7 clusters
names = DF.Zone
C7 = pd.DataFrame({"Zone": names,"group": clusters7})
A = C7.groupby(['Zone']).sum()
#print(C[C['group']==2])
print(C7)
# Effectifs par groupe :

names = DF.Zone
B = C7.groupby(['group']).count()
B = pd.DataFrame(data = B)
B.reset_index(inplace = True) #reindexation de dataframe B
B['group'] = B['group'].apply(str)
print(B)

P = pd.merge(C7, DF, how = 'inner', on = 'Zone') #jointure entre la table C et la table DF

N7 = P.groupby(['group']).mean() #Calcul de la moyenne de chaque variable pour chaque groupe

std_scale = preprocessing.StandardScaler().fit(N7)
N7_scaled = std_scale.transform(N7)
N7_scaled = pd.DataFrame(data = N7_scaled)

N7_scaled.columns = data_pca.columns
N7_scaled.index = np.arange(1, len(N7_scaled) + 1) #Commencer l'indexation de 1 au lieu de 0
N7_scaled['group'] = N7_scaled.index
N7_scaled.set_index('group',inplace=True)
N7_scaled


# In[31]:


# Heatmap avec les croisements entre les clusters de pays et les différentes variables:

N7_scaled.columns = data_pca.columns

plt.figure(figsize=(7,5))
sns.heatmap(N7_scaled, cmap = 'viridis') # Graphe carte chaleur des groupes des zones
plt.savefig("Carte chaleur avec les croisements entre les clusters de pays et les différentes variables ACH_7G.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[32]:


#Interprétations:

print(C7[C7['group']==2]) #Filtre sur le groupe des pays à recommander
#Groupe 2 : pays qui importent les plus,qui ont un bon Indice de Stabilité et qui sont assez riche donc à recommander
C7[C7['group']==2]


# In[33]:


# Projection des individus sur le 1e plan factoriel ACH_7G:

plt.figure()

plt.scatter(X_projected[:,0], X_projected[:,1],c=clusters7.astype(np.float))#Graphe nuage des points
plt.xlabel('F1')
plt.ylabel('F2')
plt.title("Projection des individus sur le 1e plan factoriel")

plt.show()


# In[34]:


# Analyse pour nombre des clusters égale à 12 :

clusters12 = fcluster(Z, 12, criterion = 'maxclust') # Coupage du dendrogramme en 12 clusters
names = DF.Zone
C12 = pd.DataFrame({"Zone": names,"group": clusters12})
A = C12.groupby(['Zone']).sum()
#print(C[C['group']==2])
print(C12)

# Effectifs par groupe :

names = DF.Zone
B = C12.groupby(['group']).count()
B = pd.DataFrame(data = B)
B.reset_index(inplace = True) #reindexation de dataframe B
B['group'] = B['group'].apply(str)
print(B)

P = pd.merge(C12, DF, how = 'inner', on = 'Zone') #jointure entre la table C et la table DF

N12 = P.groupby(['group']).mean() #Calcul de la moyenne de chaque variable pour chaque groupe

std_scale = preprocessing.StandardScaler().fit(N12)
N12_scaled = std_scale.transform(N12)
N12_scaled = pd.DataFrame(data = N12_scaled)

N12_scaled.columns = data_pca.columns
N12_scaled.index = np.arange(1, len(N12_scaled) + 1) #Commencer l'indexation de 1 au lieu de 0
N12_scaled['group'] = N12_scaled.index
N12_scaled.set_index('group',inplace=True)
N12_scaled


# In[35]:


# Heatmap avec les croisements entre les clusters de pays et les différentes variables:

N12_scaled.columns = data_pca.columns

plt.figure(figsize=(7,5))
sns.heatmap(N12_scaled, cmap = 'viridis') # Graphe carte chaleur des groupes des zones
plt.savefig("Carte chaleur avec les croisements entre les clusters de pays et les différentes variables ACH_12G.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[36]:


#Interprétations:

print(C12[C12['group']==4]) #Filtre sur le groupe des pays à recommander
#Groupe 4 : pays qui importent les plus,qui ont un bon Indice de Stabilité et qui sont assez riche donc à recommander
C12[C12['group']==4]


# In[37]:


# Projection des individus sur le 1e plan factoriel ACH_12G:

plt.figure()

plt.scatter(X_projected[:,0], X_projected[:,1],c=clusters12.astype(np.float))#Graphe nuage des points
plt.xlabel('F1')
plt.ylabel('F2')
plt.title("Projection des individus sur le 1e plan factoriel")

plt.show()


# # Kmeans

# In[38]:


inertia = []
k_range = range(1,20)
for k in k_range:
    kmeans = cluster.KMeans(n_clusters = k).fit(X_scaled)
    inertia.append(kmeans.inertia_)
kmeans.inertia_


# In[39]:


plt.plot(k_range,inertia, marker = 'o')
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Cout de model inertia")
plt.title("Elbow method")
plt.show(block = False)


# In[40]:


from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10)).fit(X_scaled)
visualizer.show()


# In[41]:


#Scatter(s) plot sur le premier plan factoriel de l'ACP avec des couleurs différentes par groupe pour # type de clustering:

fig = plt.figure(figsize=(12,12))
plt.title("Projection des {} individus sur le 1e plan factoriel pour différent nombre de groupe".format(X_projected.shape[0]))
kmeans5 = cluster.KMeans(n_clusters = 5).fit(X_scaled) # 5 clusters
ax = fig.add_subplot(321)
ax.scatter(X_projected[:,0], X_projected[:,1], c = kmeans5.labels_)#Graphe nuage des points

plt.xlabel("PC 1") #axe des abscisses
plt.ylabel("PC 2") #axe des ordonnées

kmeans6 = cluster.KMeans(n_clusters = 6).fit(X_scaled) # 6 clusters
ax = fig.add_subplot(322)
ax.scatter(X_projected[:,0], X_projected[:,1], c = kmeans6.labels_)#Graphe nuage des points

plt.xlabel("PC 1") #axe des abscisses
plt.ylabel("PC 2") #axe des ordonnées


kmeans7 = cluster.KMeans(n_clusters = 7).fit(X_scaled) # 7 clusters
ax = fig.add_subplot(323)
ax.scatter(X_projected[:,0], X_projected[:,1], c = kmeans7.labels_)#Graphe nuage des points

plt.xlabel("PC 1") #axe des abscisses
plt.ylabel("PC 2") #axe des ordonnées

kmeans8 = cluster.KMeans(n_clusters = 8).fit(X_scaled) # 8 clusters
ax = fig.add_subplot(324)
ax.scatter(X_projected[:,0], X_projected[:,1], c = kmeans8.labels_)#Graphe nuage des points

plt.xlabel("PC 1") #axe des abscisses
plt.ylabel("PC 2") #axe des ordonnées

kmeans9 = cluster.KMeans(n_clusters = 9).fit(X_scaled) # 9 clusters
ax = fig.add_subplot(325)
ax.scatter(X_projected[:,0], X_projected[:,1], c = kmeans9.labels_)#Graphe nuage des points

plt.xlabel("PC 1") #axe des abscisses
plt.ylabel("PC 2") #axe des ordonnées

kmeans10 = cluster.KMeans(n_clusters = 10).fit(X_scaled) # 10 clusters
ax = fig.add_subplot(326)
ax.scatter(X_projected[:,0], X_projected[:,1], c = kmeans10.labels_)#Graphe nuage des points

plt.xlabel("PC 1") #axe des abscisses
plt.ylabel("PC 2") #axe des ordonnées
plt.show()


# In[42]:


# Analyse pour nombre des clusters égale à 4 :

kmeans4 = cluster.KMeans(n_clusters = 4) # 4 clusters

kmeans4.fit(X_scaled)

# Récupération des clusters attribués à chaque individu

clusters = kmeans4.labels_
names = DF.Zone
C4 = pd.DataFrame({"Zone": names,"group": clusters})
A4 = C4.groupby(['Zone']).sum()

print(C4)

#Effectifs par groupe :

names = DF.Zone
B = C4.groupby(['group']).count()
B = pd.DataFrame(data = B)
B.reset_index(inplace = True) #reindexation de dataframe B
B['group'] = B['group'].apply(str)
print(B,'\n')

P = pd.merge(C4, DF, how = 'inner', on = 'Zone') #jointure entre la table C5 et la table DF

N4 = P.groupby(['group']).mean() #Calcul de la moyenne de chaque variable pour chaque groupe

std_scale = preprocessing.StandardScaler().fit(N4)
N4_scaled = std_scale.transform(N4)
N4_scaled = pd.DataFrame(data = N4_scaled)

N4_scaled.columns = data_pca.columns
N4_scaled['group'] = N4_scaled.index
N4_scaled.set_index('group',inplace=True)

print('l’inertie intraclasse :',kmeans4.inertia_)

N4_scaled #


# In[43]:


# Affichage des positions des centres de classes :
plt.figure()
centroids = kmeans4.cluster_centers_
centroids_projected = pca.transform(centroids)
plt.scatter(X_projected[:,0], X_projected[:,1], c = kmeans4.labels_)#Graphe nuage des points
plt.scatter(centroids_projected[:,0],centroids_projected[:,1],c = 'r') #Centroides
plt.title("Projection des {} centres sur le 1e plan factoriel".format(len(centroids)))
plt.show()


# In[44]:


# Heatmap avec les croisements entre les clusters de pays et les différentes variables:

N4_scaled.columns = data_pca.columns

plt.figure(figsize=(7,5))
sns.heatmap(N4_scaled, cmap = 'viridis') # Graphe carte chaleur des groupes des zones
plt.savefig("Carte chaleur avec les croisements entre les clusters de pays et les différentes variables Kmeans_4G.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[45]:


#Interprétations:

#print(C4[C4['group']==0]) #Filtre sur le groupe des pays à recommander
#Groupe 2 : pays qui importent les plus,qui ont un bon Indice de Stabilité et qui sont assez riche donc à recommander
C4[C4['group']==0]


# In[46]:


# Analyse pour nombre des clusters égale à 5 :

kmeans5 = cluster.KMeans(n_clusters = 5) # 5 clusters

kmeans5.fit(X_scaled)

# Récupération des clusters attribués à chaque individu
clusters = kmeans5.labels_
names = DF.Zone
C5 = pd.DataFrame({"Zone": names,"group": clusters})
A5 = C5.groupby(['Zone']).sum()
#print(C5[C5['group']==0])

print(C5)

#Effectifs par groupe :

names = DF.Zone
B = C5.groupby(['group']).count()
B = pd.DataFrame(data = B)
B.reset_index(inplace = True) #reindexation de dataframe B
B['group'] = B['group'].apply(str)
print(B,'\n')

P = pd.merge(C5, DF, how = 'inner', on = 'Zone') #jointure entre la table C5 et la table DF

N5 = P.groupby(['group']).mean() #Calcul de la moyenne de chaque variable pour chaque groupe

std_scale = preprocessing.StandardScaler().fit(N5)
N5_scaled = std_scale.transform(N5)
N5_scaled = pd.DataFrame(data = N5_scaled)

N5_scaled.columns = data_pca.columns
N5_scaled['group'] = N5_scaled.index
N5_scaled.set_index('group',inplace=True)

print('l’inertie intraclasse :',kmeans5.inertia_)

N5_scaled #


# In[47]:


# Affichage des positions des centres de classes :
plt.figure()
centroids = kmeans5.cluster_centers_
centroids_projected = pca.transform(centroids)
plt.scatter(X_projected[:,0], X_projected[:,1], c = kmeans5.labels_)#Graphe nuage des points
plt.scatter(centroids_projected[:,0],centroids_projected[:,1],c = 'r') #Centroides
plt.title("Projection des {} centres sur le 1e plan factoriel".format(len(centroids)))
plt.show()


# In[48]:


# Heatmap avec les croisements entre les clusters de pays et les différentes variables:

N5_scaled.columns = data_pca.columns

plt.figure(figsize=(7,5))
sns.heatmap(N5_scaled, cmap = 'viridis') # Graphe carte chaleur des groupes des zones
plt.savefig("Carte chaleur avec les croisements entre les clusters de pays et les différentes variables Kmeans_5G.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[49]:


#Interprétations:

print(C5[C5['group']==2]) #Filtre sur le groupe des pays à recommander
#Groupe 2 : pays qui importent les plus,qui ont un bon Indice de Stabilité et qui sont assez riche donc à recommander
C5[C5['group']==2]


# In[50]:


# Analyse pour nombre des clusters égale à 6 :

kmeans6 = cluster.KMeans(n_clusters = 6).fit(X_scaled) # 6 clusters

kmeans6.fit(X_scaled)

# Récupération des clusters attribués à chaque individu
clusters = kmeans6.labels_
names = DF.Zone
C6 = pd.DataFrame({"Zone": names,"group": clusters})
A6 = C6.groupby(['Zone']).sum()
#print(C5[C5['group']==0])

print(C6)

#Effectifs par groupe :

names = DF.Zone
B = C6.groupby(['group']).count()
B = pd.DataFrame(data = B)
B.reset_index(inplace = True) #reindexation de dataframe B
B['group'] = B['group'].apply(str)
print(B,'\n')

P = pd.merge(C6, DF, how = 'inner', on = 'Zone') #jointure entre la table C6 et la table DF

N6 = P.groupby(['group']).mean() #Calcul de la moyenne de chaque variable pour chaque groupe

std_scale = preprocessing.StandardScaler().fit(N6)
N6_scaled = std_scale.transform(N6)
N6_scaled = pd.DataFrame(data = N6_scaled)

N6_scaled.columns = data_pca.columns
N6_scaled['group'] = N6_scaled.index
N6_scaled.set_index('group',inplace=True)

print('l’inertie intraclasse :',kmeans6.inertia_)

N6_scaled #


# In[51]:


# Affichage des positions des centres de classes
plt.figure()
centroids = kmeans6.cluster_centers_
centroids_projected = pca.transform(centroids)
plt.scatter(X_projected[:,0], X_projected[:,1], c = kmeans6.labels_) #Graphe nuage des points 
plt.scatter(centroids_projected[:,0],centroids_projected[:,1], c = 'r') #Centroides
plt.title("Projection des {} centres sur le 1e plan factoriel".format(len(centroids)))
plt.show()


# In[52]:


centroids


# In[53]:


# Heatmap avec les croisements entre les clusters de pays et les différentes variables:

N6_scaled.columns = data_pca.columns

plt.figure(figsize=(7,5))
sns.heatmap(N6_scaled, cmap = 'viridis') # Graphe carte chaleur des groupes des zones
plt.savefig("Carte chaleur avec les croisements entre les clusters de pays et les différentes variables Kmeans_6G.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[54]:


#Interprétations:

print(C6[C6['group']==5]) #Filtre sur le groupe des pays à recommander
#Groupe 5 : pays qui importent les plus,qui ont un bon Indice de Stabilité et qui sont assez riche donc à recommander
C6[C6['group']==5]


# In[55]:


# Analyse pour nombre des clusters égale à 7 :

kmeans7 = cluster.KMeans(n_clusters = 7).fit(X_scaled) # 7 clusters

kmeans7.fit(X_scaled)

# Récupération des clusters attribués à chaque individu
clusters = kmeans7.labels_
names = DF.Zone
C7 = pd.DataFrame({"Zone": names,"group": clusters})
A7 = C7.groupby(['Zone']).sum()
#print(C5[C5['group']==0])

print(C7)

#Effectifs par groupe :

names = DF.Zone
B = C7.groupby(['group']).count()
B = pd.DataFrame(data = B)
B.reset_index(inplace = True) #reindexation de dataframe B
B['group'] = B['group'].apply(str)
print(B,'\n')

P = pd.merge(C7, DF, how = 'inner', on = 'Zone') #jointure entre la table C7 et la table DF

N7 = P.groupby(['group']).mean() #Calcul de la moyenne de chaque variable pour chaque groupe

std_scale = preprocessing.StandardScaler().fit(N7)
N7_scaled = std_scale.transform(N7)
N7_scaled = pd.DataFrame(data = N7_scaled)

N7_scaled.columns = data_pca.columns
N7_scaled['group'] = N7_scaled.index
N7_scaled.set_index('group',inplace=True)
print('l’inertie intraclasse :',kmeans7.inertia_)

N7_scaled #


# In[56]:


# Heatmap avec les croisements entre les clusters de pays et les différentes variables:

N7_scaled.columns = data_pca.columns

plt.figure(figsize=(7,5))
sns.heatmap(N7_scaled, cmap = 'viridis') # Graphe carte chaleur des groupes des zones
plt.savefig("Carte chaleur avec les croisements entre les clusters de pays et les différentes variables Kmeans_7G.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[57]:


#Interprétations:

print(C7[C7['group']==6]) #Filtre sur le groupe des pays à recommander
#Groupe 4 : pays qui importent les plus,qui ont un bon Indice de Stabilité et qui sont assez riche donc à recommander
C7[C7['group']==6]


# In[58]:


# Affichage des positions des centres de classes
plt.figure()
centroids = kmeans7.cluster_centers_
centroids_projected = pca.transform(centroids)
plt.scatter(X_projected[:,0], X_projected[:,1], c = kmeans7.labels_)#Graphe nuage des points
plt.scatter(centroids_projected[:,0],centroids_projected[:,1], c='r')#Centroides
plt.title("Projection des {} centres sur le 1e plan factoriel".format(len(centroids)))
plt.show()


# In[59]:


# Analyse pour nombre des clusters égale à 8 :

kmeans8 = cluster.KMeans(n_clusters = 8).fit(X_scaled) # 8 clusters

kmeans8.fit(X_scaled)

# Récupération des clusters attribués à chaque individu
clusters = kmeans8.labels_
names = DF.Zone
C8 = pd.DataFrame({"Zone": names,"group": clusters})
A8 = C8.groupby(['Zone']).sum()
#print(C5[C5['group']==0])

print(C8)

#Effectifs par groupe :

names = DF.Zone
B = C8.groupby(['group']).count()
B = pd.DataFrame(data = B)
B.reset_index(inplace = True) #reindexation de dataframe B
B['group'] = B['group'].apply(str)
print(B,'\n')

P = pd.merge(C8, DF, how = 'inner', on = 'Zone') #jointure entre la table C6 et la table DF

N8 = P.groupby(['group']).mean() #Calcul de la moyenne de chaque variable pour chaque groupe

std_scale = preprocessing.StandardScaler().fit(N8)
N8_scaled = std_scale.transform(N8)
N8_scaled = pd.DataFrame(data = N8_scaled)

N8_scaled.columns = data_pca.columns
N8_scaled['group'] = N8_scaled.index
N8_scaled.set_index('group',inplace=True)
print('l’inertie intraclasse :',kmeans8.inertia_)

N8_scaled #


# In[60]:


# Affichage des positions des centres de classes
plt.figure()
centroids = kmeans8.cluster_centers_
centroids_projected = pca.transform(centroids)
plt.scatter(X_projected[:,0], X_projected[:,1], c = kmeans8.labels_)#Graphe nuage des points
plt.scatter(centroids_projected[:,0],centroids_projected[:,1], c = 'r')#Centroides
plt.title("Projection des {} centres sur le 1e plan factoriel".format(len(centroids)))
plt.show()


# In[61]:


# Heatmap avec les croisements entre les clusters de pays et les différentes variables:

N8_scaled.columns = data_pca.columns

plt.figure(figsize=(7,5))
sns.heatmap(N8_scaled, cmap = 'viridis') # Graphe carte chaleur des groupes des zones
plt.savefig("Carte chaleur avec les croisements entre les clusters de pays et les différentes variables Kmeans_8G.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[62]:


#Interprétations:

print(C8[C8['group']==7]) #Filtre sur le groupe des pays à recommander
#Groupe 4 : pays qui importent les plus,qui ont un bon Indice de Stabilité et qui sont assez riche donc à recommander


# In[63]:


# Analyse pour nombre des clusters égale à 9 :

kmeans9 = cluster.KMeans(n_clusters = 9).fit(X_scaled) # 9 clusters

kmeans9.fit(X_scaled)

# Récupération des clusters attribués à chaque individu
clusters = kmeans9.labels_
names = DF.Zone
C9 = pd.DataFrame({"Zone": names,"group": clusters})
A9 = C9.groupby(['Zone']).sum()
#print(C5[C5['group']==0])

print(C9)

#Effectifs par groupe :

names = DF.Zone
B = C9.groupby(['group']).count()
B = pd.DataFrame(data = B)
B.reset_index(inplace = True) #reindexation de dataframe B
B['group'] = B['group'].apply(str)
print(B,'\n')

P = pd.merge(C9, DF, how = 'inner', on = 'Zone') #jointure entre la table C9 et la table DF

N9 = P.groupby(['group']).mean() #Calcul de la moyenne de chaque variable pour chaque groupe

std_scale = preprocessing.StandardScaler().fit(N9)
N9_scaled = std_scale.transform(N9)
N9_scaled = pd.DataFrame(data = N9_scaled)

N9_scaled.columns = data_pca.columns
N9_scaled['group'] = N9_scaled.index
N9_scaled.set_index('group',inplace=True)
print('l’inertie intraclasse :',kmeans9.inertia_)

N9_scaled #


# In[64]:


# Affichage des positions des centres de classes
plt.figure()
centroids = kmeans9.cluster_centers_
centroids_projected = pca.transform(centroids)
plt.scatter(X_projected[:,0], X_projected[:,1], c = kmeans9.labels_)#Graphe nuage des points
plt.scatter(centroids_projected[:,0],centroids_projected[:,1], c = 'r')#Centroides
plt.title("Projection des {} centres sur le 1e plan factoriel".format(len(centroids)))
plt.show()


# In[65]:


# Heatmap avec les croisements entre les clusters de pays et les différentes variables:

N9_scaled.columns = data_pca.columns

plt.figure(figsize=(7,5))
sns.heatmap(N9_scaled, cmap = 'viridis') # Graphe carte chaleur des groupes des zones
plt.savefig("Carte chaleur avec les croisements entre les clusters de pays et les différentes variables Kmeans_9G.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[66]:


#Interprétations:

print(C9[C9['group']==6]) #Filtre sur le groupe des pays à recommander
#Groupe 2 : pays qui importent les plus,qui ont un bon Indice de Stabilité et qui sont assez riche donc à recommander


# In[71]:


# Analyse pour nombre des clusters égale à 10 :

kmeans10 = cluster.KMeans(n_clusters = 10).fit(X_scaled) # 10 clusters

#kmeans10.fit(X_scaled)

# Récupération des clusters attribués à chaque individu
clusters = kmeans10.labels_
names = DF.Zone
C10 = pd.DataFrame({"Zone": names,"group": clusters})
A10 = C10.groupby(['Zone']).sum()
#print(C5[C5['group']==0])

print(C10)

#Effectifs par groupe :

names = DF.Zone
B = C10.groupby(['group']).count()
B = pd.DataFrame(data = B)
B.reset_index(inplace = True) #reindexation de dataframe B
B['group'] = B['group'].apply(str)
print(B,'\n')

P = pd.merge(C10, DF, how = 'inner', on = 'Zone') #jointure entre la table C10 et la table DF

N10 = P.groupby(['group']).mean() #Calcul de la moyenne de chaque variable pour chaque groupe

std_scale = preprocessing.StandardScaler().fit(N10)
N10_scaled = std_scale.transform(N10)
N10_scaled = pd.DataFrame(data = N10_scaled)

N10_scaled.columns = data_pca.columns
N10_scaled['group'] = N10_scaled.index
N10_scaled.set_index('group',inplace=True)
print('l’inertie intraclasse :',kmeans10.inertia_)

N10_scaled #


# In[72]:


# Affichage des positions des centres de classes
plt.figure()
centroids = kmeans10.cluster_centers_
centroids_projected = pca.transform(centroids)
plt.scatter(X_projected[:,0], X_projected[:,1], c = kmeans10.labels_)#Graphe nuage des points
plt.scatter(centroids_projected[:,0],centroids_projected[:,1], c = 'r') #Centroides
plt.title("Projection des {} centres sur le 1e plan factoriel".format(len(centroids)))
plt.show()


# In[69]:


# Heatmap avec les croisements entre les clusters de pays et les différentes variables:

N10_scaled.columns = data_pca.columns

plt.figure(figsize=(7,5))
sns.heatmap(N10_scaled, cmap = 'viridis') # Graphe carte chaleur des groupes des zones
plt.savefig("Carte chaleur avec les croisements entre les clusters de pays et les différentes variables Kmeans_10G.png" , dpi = 300,bbox_inches = 'tight') #enregistrement de figure dans le répertoire de travail et augmentation de la résolution
plt.show()


# In[70]:


#Interprétations:

print(C10[C10['group']==5]) #Filtre sur le groupe des pays à recommander
#Groupe 8 : pays qui importent les plus,qui ont un bon Indice de Stabilité et qui sont assez riche donc à recommander
print(C10[C10['group']==9]) #Filtre sur le groupe des pays à recommander
#Groupe 8 : pays qui importent les plus,qui ont un bon Indice de Stabilité et qui sont assez riche donc à recommander

