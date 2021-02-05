#####
## Noah Serr V00891494
## SENG 474 A1
## Decision Trees
#####

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


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


# Implementation of decision tree with pruning.
def decision_tree(X_train, X_test, y_train, y_test, criterion, max_features):
    test_scores, training_scores = [], []
    trees, nodes = [], []

    clf = DecisionTreeClassifier(random_state = 0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    for ccp_alpha in ccp_alphas:
        clf = DecisionTreeClassifier(ccp_alpha = ccp_alpha, criterion = criterion, max_features = max_features)
        clf = clf.fit(X_train, y_train)

        y_train_prediction = clf.predict(X_train)
        y_test_prediction = clf.predict(X_test)
        
        if clf.tree_.node_count not in nodes:
            training_scores.append(metrics.accuracy_score(y_train, y_train_prediction))
            test_scores.append(metrics.accuracy_score(y_test, y_test_prediction))
            
            trees.append(clf)
            nodes.append(clf.tree_.node_count)

    return max(test_scores)
    

# Use matplotlib to create graphs for results.
def generate_plot(title, xlabel, ylabel, data, name):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.plot(data, 'r', label = "20% Test Split")
    plt.legend()
    
    filename = name + ".png"
    plt.savefig(filename)
    plt.clf()


def main():
    X_train_heart, X_test_heart, y_train_heart, y_test_heart = heart_data_setup()
    X_train_seeds, X_test_seeds, y_train_seeds, y_test_seeds = seeds_data_setup()

    criterion = 'entropy'
    heart_trees = []
    seeds_trees = []
    heart_max = 13
    seeds_max = 7

    for max_features in range(1, heart_max):
        heart_trees.append(decision_tree(X_train_heart, X_test_heart, y_train_heart, y_test_heart, criterion, max_features))

    for max_features in range(1, seeds_max):
        seeds_trees.append(decision_tree(X_train_seeds, X_test_seeds, y_train_seeds, y_test_seeds, criterion, max_features))

    generate_plot("Heart Disease Data (Decision Trees / Gini)", "Number of Features", "Accuracy", heart_trees, "HeartDTGini")
    generate_plot("Wheet Seeds Data (Decision Trees / Entropy)", "Number of Features", "Accuracy", seeds_trees, "SeedsDTEntropy")


main()