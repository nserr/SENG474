#####
## Noah Serr V00891494
## SENG 474 A2
## Logistic Regression
#####

import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import mnist_reader


# Reads, initializes, and converts data to binary classification problem.
def init():
    X_train, y_train = mnist_reader.load_mnist('fashion-mnist-master/data/fashion', kind = 'train')
    X_test, y_test = mnist_reader.load_mnist('fashion-mnist-master/data/fashion', kind = 't10k')

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train_temp, y_train_temp = [], []
    X_test_temp, y_test_temp = [], []

    train_len = len(X_train)
    test_len = len(X_test)

    # Set training data as binary classification problem.
    for i in range(train_len):

        if y_train[i] == 5:
            y_train_temp.append(0)
            X_train_temp.append(X_train[i])

        if y_train[i] == 7:
            y_train_temp.append(1)
            X_train_temp.append(X_train[i])

    # Set test data as binary classification problem.
    for i in range(test_len):

        if y_test[i] == 5:
            y_test_temp.append(0)
            X_test_temp.append(X_test[i])

        if y_test[i] == 7:
            y_test_temp.append(1)
            X_test_temp.append(X_test[i])

    X_train = np.array(X_train_temp) / 255
    y_train = y_train_temp
    X_test = np.array(X_test_temp) / 255
    y_test = y_test_temp

    return X_train, y_train, X_test, y_test


# Implementation of logistic regression.
def log_reg(X_train, y_train, X_test, y_test, c):
    clf = LogisticRegression(penalty = 'l2', max_iter = 1000000, C = c)
    clf.fit(X_train, y_train)

    train_accuracy = clf.score(X_train, y_train)
    y_test_prediction = clf.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_test, y_test_prediction)

    return train_accuracy, test_accuracy


# Generates regulization parameters using a logarithmically spaced grid.
def generate_c_values(initial, alpha):
    c_values = []

    for i in range(0, 10):
        c_values.append(initial * (alpha ** i))
    
    return c_values


# Use matplotlib to create graph for results.
def generate_plot(train_accuracies, test_accuracies, c_values):
    plt.title("Logistic Regression")
    plt.xlabel("Regularization Parameter")
    plt.xscale('log')
    plt.ylabel("Accuracy")

    plt.plot(c_values, train_accuracies, 'r', label = "Train Accuracy")
    plt.plot(c_values, test_accuracies, 'b', label = "Test Accuracy")

    plt.legend()

    filename = "LogisticRegression.png"
    plt.savefig(filename)
    plt.clf()


def main():
    X_train, y_train, X_test, y_test = init()
    c_values = generate_c_values(0.000001, 10)

    train_accuracies = []
    test_accuracies = []

    for c in c_values:
        train_accuracy, test_accuracy = log_reg(X_train, y_train, X_test, y_test, c)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    generate_plot(train_accuracies, test_accuracies, c_values)
    

main()