#####
## Noah Serr V00891494
## SENG 474 A3
## Hierarchical Agglomerative Clustering
#####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering as ac


# Implementation of Single Linkage Hierarchical Agglomerative Clustering.
def single_linkage(dataset):
    path = "data/" + dataset + ".csv"
    df = pd.read_csv(path)

    if dataset == 'dataset1':
        x = df.iloc[:, [0, 1]].values
        d = '2D'
    else:
        x = df.iloc[:, [0, 1, 2]].values
        d = '3D'

    hac = ac(n_clusters = None, distance_threshold = 1, linkage = 'single')
    hac.fit(x)

    linkages = create_dendogram(hac)
    dendrogram(linkages, truncate_mode = 'lastp')

    plt.title("Dendogram for Single Linkage HAC with " + dataset)
    filename = "singleHAC" + d + "_dendogram"
    plt.savefig(filename)
    plt.clf()

    if d == '2D':
        generate_2D_plot(x, 28, 'single')
    else:
        generate_3D_plot(x, 28, 'single')


# Implementation of Average Linkage Hierarchical Agglomerative Clustering.
def average_linkage(dataset):
    path = "data/" + dataset + ".csv"
    df = pd.read_csv(path)

    if dataset == 'dataset1':
        x = df.iloc[:, [0, 1]].values
        d = '2D'
    else:
        x = df.iloc[:, [0, 1, 2]].values
        d = '3D'
    
    hac = ac(n_clusters = None, distance_threshold = 1, linkage = 'average')
    hac.fit(x)

    linkages = create_dendogram(hac)
    dendrogram(linkages, truncate_mode = 'lastp')

    plt.title("Dendogram for Average Linkage HAC with " + dataset)
    filename = "averageHAC" + d + "_dendogram"
    plt.savefig(filename)
    plt.clf()

    if d == '2D':
        generate_2D_plot(x, 3, 'average')
    else:
        generate_3D_plot(x, 26, 'average')


# Function to create dendograms.
def create_dendogram(hac):
    n = len(hac.labels_)
    counts = np.zeros(hac.children_.shape[0])

    for i, j in enumerate(hac.children_):
        count = 0
        for child in j:
            if child < n:
                count += 1
            else:
                count += counts[child - n]
        
        counts[i] = count

    linkages = np.column_stack([hac.children_, hac.distances_, counts])
    linkages = linkages.astype(float)

    return linkages


# Function to create cluster graphs with 2D dataset.
def generate_2D_plot(x, clusters, linkage):
    hac = ac(n_clusters = clusters, affinity = 'euclidean', linkage = linkage)
    hac.fit_predict(x)

    plt.title("Cluster Map for " + linkage.capitalize() + " Linkage HAC with dataset1")
    plt.scatter(x[:, 0], x[:, 1], c = hac.labels_)
    filename = linkage + "HAC2D_cluster"
    plt.savefig(filename)
    plt.clf()


# Function to create cluster graphs with 3D dataset.
def generate_3D_plot(x, clusters, linkage):
    hac = ac(n_clusters = clusters, affinity = 'euclidean', linkage = linkage)
    hac.fit_predict(x)

    fig = plt.figure()
    ax = Axes3D(fig)

    ax.set_title("Cluster Map for " + linkage.capitalize() + " Linkage HAC with dataset2")
    ax.scatter(x[:, 0], x[:, 1], x[:, 2], c = hac.labels_)
    filename = linkage + "HAC3D_cluster"
    plt.savefig(filename)
    plt.clf()


def main():
    single_linkage('dataset1')
    single_linkage('dataset2')

    average_linkage('dataset1')
    average_linkage('dataset2')

main()