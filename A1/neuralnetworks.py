#####
## Noah Serr V00891494
## SENG 474 A1
## Neural Networks
#####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


# Setup heart disease data. Columnn names from the data description.
def heart_data_setup():
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    heart_data = pd.read_csv("./data/cleaned_processed.cleveland.data", header = None, names = columns)

    X = heart_data[columns[:13]] # All columns aside from the predicted attribute.
    y = heart_data.num # The predicted attribute.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # 20% test split size
    return X_train, X_test, y_train, y_test


# Setup wheat seeds data. Column names from the data description.
def seeds_data_setup():
    columns = ['area', 'perim', 'compact', 'length', 'width', 'asym', 'length_groove', 'target']
    seeds_data = pd.read_csv("./data/seeds_dataset.data", header = None, names = columns)

    X = seeds_data[columns[:7]]
    y = seeds_data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    return X_train, X_test, y_train, y_test


# Implementation of neural networks.
def neural_networks(X_train, X_test, y_train, y_test, hidden_layer_sizes, max_iter, learning_rate_init):
    clf = MLPClassifier(hidden_layer_sizes = hidden_layer_sizes, max_iter = max_iter, learning_rate_init = learning_rate_init)
    clf.fit(X_train, y_train)

    y_train_prediction = clf.predict(X_train)
    y_test_prediction = clf.predict(X_test)

    return metrics.accuracy_score(y_test, y_test_prediction)


# Use matplotlib to create graphs for results.
def generate_plot(title, xlabel, ylabel, data, name):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.plot(data, 'b', label = "0.1 Learning Rate")
    plt.legend()
    
    filename = name + ".png"
    plt.savefig(filename)
    plt.clf()


def main():
    X_train_heart, X_test_heart, y_train_heart, y_test_heart = heart_data_setup()
    X_train_seeds, X_test_seeds, y_train_seeds, y_test_seeds = seeds_data_setup()

    hidden_layer_sizes = [5, 10, 50]
    max_iter = [1, 10, 20]
    learning_rate_init = 0.1

    heart_networks_5 = []
    seeds_networks_5 = []
    heart_networks_10 = []
    seeds_networks_10 = []
    heart_networks_50 = []
    seeds_networks_50 = []

    for layer in hidden_layer_sizes:
        heart_networks = []
        seeds_networks = []

        for num_iter in max_iter:
            for iteration in range(1, num_iter):
                heart_networks.append(neural_networks(X_train_heart, X_test_heart, y_train_heart, y_test_heart, layer, iteration, learning_rate_init))
                seeds_networks.append(neural_networks(X_train_seeds, X_test_seeds, y_train_seeds, y_test_seeds, layer, iteration, learning_rate_init))

        if layer == 5:
            heart_networks_5 = heart_networks
            seeds_networks_5 = seeds_networks
        elif layer == 10:
            heart_networks_10 = heart_networks
            seeds_networks_10 = seeds_networks
        elif layer == 50:
            heart_networks_50 = heart_networks
            seeds_networks_50 = seeds_networks
    
    generate_plot("Heart Disease Data (Neural Networks / 5 Hidden Layers)", "Number of Iterations", "Accuracy", heart_networks_5, "HeartNN5")
    generate_plot("Wheat Seeds Data (Neural Networks / 5 Hidden Layers)", "Number of Iterations", "Accuracy", seeds_networks_5, "SeedsNN5")

    generate_plot("Heart Disease Data (Neural Networks / 10 Hidden Layers)", "Number of Iterations", "Accuracy", heart_networks_10, "HeartNN10")
    generate_plot("Wheat Seeds Data (Neural Networks / 10 Hidden Layers)", "Number of Iterations", "Accuracy", seeds_networks_10, "SeedsNN10")

    generate_plot("Heart Disease Data (Neural Networks / 50 Hidden Layers)", "Number of Iterations", "Accuracy", heart_networks_50, "HeartNN50")
    generate_plot("Wheat Seeds Data (Neural Networks / 50 Hidden Layers)", "Number of Iterations", "Accuracy", seeds_networks_50, "SeedsNN50")


main()