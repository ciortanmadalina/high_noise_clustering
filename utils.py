import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pickle
import random
from tqdm import tqdm
import numpy as np
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from IPython.display import clear_output, Image, display
from sklearn.datasets.samples_generator import make_blobs
import itertools
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import kneighbors_graph
import igraph as ig
import louvain
from sklearn.metrics.cluster import adjusted_rand_score
import umap
import os
from scipy import sparse, io

plt.ion()
plt.show()

printFunctionNames = True

if printFunctionNames:
    print('elbowAnalysis')
def elbowAnalysis(X, numberOfClusters):
    distortions = []

    for k in tqdm(numberOfClusters):
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    plt.plot(numberOfClusters, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    
    
if printFunctionNames:
    print('silhouetteAnalyis')
def silhouetteAnalyis (X, numberOfClusters):
    silhouette_score_values=[]
    for i in tqdm(numberOfClusters):
        classifier=KMeans(i,init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True)
        classifier.fit(X)
        labels= classifier.predict(X)
        silhouette_score_values.append(metrics.silhouette_score(X,labels ,metric='euclidean', sample_size=None, random_state=None))

    plt.plot(numberOfClusters, silhouette_score_values)
    plt.title("Silhouette score values vs Numbers of Clusters ")
    plt.show()

    Optimal_NumberOf_Components=numberOfClusters[silhouette_score_values.index(max(silhouette_score_values))]
    print( "Optimal number of components is:", Optimal_NumberOf_Components)
    
    
def loadData(inputDataset):
    """
    Load input dataset
    """
    if inputDataset == 'brainCIDR':
        path = '../input/brainCIDR/'
        df = pd.read_csv(f"{path}brainTags.csv", index_col = 0).T
        truth = pd.read_pickle(f'{path}truth.pkl')
    
    if inputDataset == 'pancreaticIsletCIDR':
        path = '../input/pancreaticIsletCIDR/'
        df = pd.read_csv(f"{path}pancreaticIsletTags.csv", index_col = 0).T
        truth = pd.read_pickle(f'{path}truth.pkl')
    
    if inputDataset == 'deng':
        path = '../input/deng/'
        df = pd.read_csv(f"{path}deng.csv", index_col = 0).T
        truth = pd.read_pickle(f'{path}truth.pkl')
        
    if inputDataset == 'celegans':
        path = '../input/celengans/'
        data = sparse.load_npz(f"{path}sparse_data.npz")
        data = data.todense()
        df1 = pd.DataFrame(data = data)
        df1.set_index(np.load(f"{path}cells.npy"), inplace=True)
        df1.columns = np.load(f"{path}genes.npy")
        return df1, None
    
    if inputDataset in [ 'sce10x_qc', 'sce2_qc', 'sce8_qc']:
        path = '../input/cellBench/'
        data = sparse.load_npz(f"{path}{inputDataset}.npz")
        data = data.todense()
        df = pd.DataFrame(data = data)
        df.set_index(np.load(f"{path}{inputDataset}_cells.npy"), inplace=True)
        df.columns = np.load(f"{path}{inputDataset}_genes.npy")
        truth = pd.read_pickle(f'{path}{inputDataset}_truth.pkl')
        
    return df, truth



def externalValidation(truthClusters, predictedClusters):
    def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
    scores = {}
    scores['_rand_index'] = adjusted_rand_score(truthClusters, predictedClusters)
    scores['_homogeneity_score'] = metrics.homogeneity_score(truthClusters, predictedClusters)
    scores['_purity_score'] = purity_score(truthClusters, predictedClusters)
    scores['_adjusted_mutual_info_score'] = metrics.adjusted_mutual_info_score(truthClusters, predictedClusters)
    scores['_fowlkes_mallows_score'] = metrics.fowlkes_mallows_score(truthClusters, predictedClusters)  
    return scores


def internalValidation(data, clusters):
    scores = {}
    """
    The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. 
    Scores around zero indicate overlapping clusters.
    The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    """
    scores['_silhouette_score'] =metrics.silhouette_score(data,clusters ,metric='euclidean')
    """
    The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
    The score is fast to compute
    """
    scores['_calinski_harabaz_score'] = metrics.calinski_harabaz_score(data,clusters)
    """
    Zero is the lowest possible score. Values closer to zero indicate a better partition.
    The Davies-Boulding index is generally higher for convex clusters than other concepts of clusters, 
    such as density based clusters like those obtained from DBSCAN.
    """
    scores['_davies_bouldin_score'] = metrics.davies_bouldin_score(data,clusters)
    return scores


def plotCorrelation(resultsDf, name = None):
    import seaborn as sns
    scoreColumns = [c for c in resultsDf.columns if c.startswith('_')]
    score = resultsDf[scoreColumns]
    score = score.astype(float)
    sns.set(font_scale=0.9)
#     fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(score.corr(), annot=True);
    if name is not None:
        plt.title(f"Correlation for {name}")