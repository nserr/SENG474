#####
## Noah Serr V00891494
## SENG 474 A3
## Lloyd's Algorithm (k-means)
#####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rand
from mpl_toolkits.mplot3d import Axes3D


def uniform_random_init(centroids, k, m, x):
    for n in range(k):
        r = rand.randint(0, m - 1)
        centroids = np.c_[centroids, x[r]]

    return centroids


def kmeans_pp_init():
    return


def kmeans_2d(init_method):
    df = pd.read_csv('data/dataset1.csv')

    x = df.iloc[:, [0, 1]].values
    m = x.shape[0]

    arr = np.array([])

    iters = 10
    method = ''

    for k in range(2, 11, 2):
        output = {}
        centroids = np.array([]).reshape(x.shape[1], 0)

        if (init_method == 'URI'):
            centroids = uniform_random_init(centroids, k, m, x)
            method = "Uniform Random Initialization"
        elif (init_method == 'K++'):
            # centroids = kmeans_pp_init()
            method = "K-Means++ Initialization"

        for i in range(iters):
            dists = np.array([]).reshape(m, 0)

            for n in range(k):
                result = (x - centroids[:, n])**2
                dist = np.sum(result, axis = 1)
                dists = np.c_[dists, dist]

            cluster = np.argmin(dists, axis = 1) + 1

            for n in range(k):
                output[n + 1] = np.array([]).reshape(2, 0)

            for n in range(m):
                output[cluster[n]] = np.c_[output[cluster[n]], x[n]]

            for n in range(k):
                output[n + 1] = output[n + 1].T
                centroids[:, n] = np.mean(output[n + 1], axis = 0)

        centroids = centroids.T
        sum_se = 0

        for n in range(k):
            se = (output[n + 1] - centroids[n, :])**2
            sum_se += np.sum(se)
        
        arr = np.append(arr, sum_se)
        
        for n in range(k):
            plt.scatter(output[n + 1][:, 0], output[n + 1][:, 1])
        
        plt.title("K-Means (2D, " + method + ") with " + str(k) + " Clusters")
        plt.scatter(centroids[:, 0], centroids[:, 1], s = 300, c = 'blue', label = 'Centroids')
        filename = "kmeans2d_" + init_method + "_" + str(k)
        plt.savefig(filename)
        plt.clf()

    return


def kmeans_3d():
    return


def main():
    kmeans_2d('URI')

main()