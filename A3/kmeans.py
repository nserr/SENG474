#####
## Noah Serr V00891494
## SENG 474 A3
## Lloyd's Algorithm (K-Means)
#####

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random as rand
from mpl_toolkits.mplot3d import Axes3D


# Implementation of Uniform Random Initialization.
def uniform_random_init(centroids, k, x):
    for n in range(k):
        r = rand.randint(0, x.shape[0] - 1)
        centroids = np.c_[centroids, x[r]]

    return centroids


# Implementation of K-Means++ Initialization.
def kmeans_pp_init(x, k):
    r = rand.randint(0, x.shape[0])
    centroid = np.array([x[r]])

    for n in range(1, k):
        r = 0
        r2 = rand.random()
        arr = np.array([])

        for cur_x in x:
            result = np.min(np.sum((cur_x - centroid)**2))
            arr = np.append(arr, result)

        prob = arr / np.sum(arr)
        cum_prob = np.cumsum(prob)

        for i, j in enumerate(cum_prob):
            if r2 < j:
                r = i
                break
        
        centroid = np.append(centroid, [x[r]], axis = 0)

    return centroid.T


# Implementation of K-Means Algorithm to deal with 2D data (Dataset 1).
def kmeans_2d(init_method):
    df = pd.read_csv('data/dataset1.csv')

    x = df.iloc[:, [0, 1]].values
    m = x.shape[0]

    sum_se_arr = np.array([])

    iters = 10
    method = ''

    for k in range(2, 11, 2):
        output = {}
        centroids = np.array([]).reshape(x.shape[1], 0)

        if (init_method == 'URI'):
            centroids = uniform_random_init(centroids, k, x)
            method = "Uniform Random Initialization"
        elif (init_method == 'K++'):
            centroids = kmeans_pp_init(x, k)
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
        
        sum_se_arr = np.append(sum_se_arr, sum_se)
        
        for n in range(k):
            plt.scatter(output[n + 1][:, 0], output[n + 1][:, 1])
        
        plt.title("K-Means (2D, " + method + ") with " + str(k) + " Clusters")
        plt.scatter(centroids[:, 0], centroids[:, 1], s = 300, c = 'blue', label = 'Centroids')
        filename = "kmeans2d_" + init_method + "_" + str(k)
        plt.savefig(filename)
        plt.clf()

    plot_cost(sum_se_arr, '2D', init_method)



# Implementation of K-Means Algorithm to deal with 3D data (Dataset 2).
def kmeans_3d(init_method):
    df = pd.read_csv('data/dataset2.csv')

    x = df.iloc[:, [0, 1, 2]].values
    m = x.shape[0]

    sum_se_arr = np.array([])
    
    iters = 10
    method = ''

    for k in range(2, 11, 2):
        output = {}
        centroids = np.array([]).reshape(x.shape[1], 0)

        if (init_method == 'URI'):
            centroids = uniform_random_init(centroids, k, x)
            method = "Uniform Random Initialization"
        elif (init_method == 'K++'):
            centroids = kmeans_pp_init(x, k)
            method = "K-Means++ Initialization"
        
        for i in range(iters):
            dists = np.array([]).reshape(m, 0)

            for n in range(k):
                result = (x - centroids[:, n])**2
                dist = np.sum(result, axis = 1)
                dists = np.c_[dists, dist]

            cluster = np.argmin(dists, axis = 1) + 1

            for n in range(k):
                output[n + 1] = np.array([]).reshape(3, 0)

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

        sum_se_arr = np.append(sum_se_arr, sum_se)

        fig = plt.figure()
        ax = Axes3D(fig)

        for n in range(k):
            ax.scatter(output[n + 1][:, 0], output[n + 1][:, 1], output[n + 1][:, 2])
        
        ax.set_title("K-Means (3D, " + method + ") with " + str(k) + " Clusters")
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], s = 300, c = 'blue', label = 'Centroids')
        filename = "kmeans3d_" + init_method + "_" + str(k)
        plt.savefig(filename)
        plt.clf()

    plot_cost(sum_se_arr, '3D', init_method)


# Function to create graphs of cost as k increases.
def plot_cost(sum_se_arr, dataset, method):
    clusters = [2, 4, 6, 8, 10]

    plt.title("Cost for " + dataset + " Dataset Using " + method)
    plt.plot(clusters, sum_se_arr, color = 'blue')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Sum of Squared Errors")

    filename = "cost_" + dataset + "_" + method
    plt.savefig(filename)
    plt.clf()


def main():
    ### K-Means ###
    # Takes initialization method as argument. 'URI' or 'K++'.
    kmeans_2d('URI')
    kmeans_3d('K++') 


main()