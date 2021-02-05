#####
## Noah Serr V00891494
## SENG 474 A1
## Random Forests
#####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


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


# Implementation of random forest.
def random_forest(X_train, X_test, y_train, y_test, criterion, n_estimators, max_features):
    clf = RandomForestClassifier(criterion = criterion, n_estimators = n_estimators, max_features = max_features)
    clf.fit(X_train, y_train)

    y_train_prediction = clf.predict(X_train)
    y_test_prediction = clf.predict(X_test)

    return metrics.accuracy_score(y_train, y_train_prediction)


# Use matplotlib to create graphs for results.
def generate_plot(title, xlabel, ylabel, data, name):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.plot(data[0], 'r', label = "1 Tree")
    plt.plot(data[1], 'g', label = "10 Trees")
    plt.plot(data[2], 'b', label = "100 Trees")
    
    plt.legend()
    
    filename = name + ".png"
    plt.savefig(filename)
    plt.clf()


def main():
    X_train_heart, X_test_heart, y_train_heart, y_test_heart = heart_data_setup()
    X_train_seeds, X_test_seeds, y_train_seeds, y_test_seeds = seeds_data_setup()

    criterion = 'gini'
    n_estimators = [1, 10, 100]

    heart_forests_n = []
    seeds_forests_n = []

    heart_max = 13
    seeds_max = 7

    for n in n_estimators:
        heart_forests = []
        seeds_forests = []

        for max_features in range(1, heart_max):
            heart_forests.append(random_forest(X_train_heart, X_test_heart, y_train_heart, y_test_heart, criterion, n, max_features))
        
        for max_features in range(1, seeds_max):
            seeds_forests.append(random_forest(X_train_seeds, X_test_seeds, y_train_seeds, y_test_seeds, criterion, n, max_features))

        heart_forests_n.append(heart_forests)
        seeds_forests_n.append(seeds_forests)

    generate_plot("Heart Disease Data (Random Forests / Gini)", "Number of Features", "Accuracy", heart_forests_n, "HeartRFGini")
    generate_plot("Wheat Seeds Data (Random Forests / Gini)", "Number of Features", "Accuracy", seeds_forests_n, "SeedsRFGini")


main()