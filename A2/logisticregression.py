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

    sandal_count = 0
    sneaker_count = 0

    # Convert training data to binary classification problem.
    for i in range(train_len):

        if y_train[i] == 5 and sandal_count < 1000:
            y_train_temp.append(0)
            X_train_temp.append(X_train[i])

        if y_train[i] == 7 and sneaker_count < 1000:
            y_train_temp.append(1)
            X_train_temp.append(X_train[i])

    # Convert test data to binary classification problem.
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
def logistic_regression(X_train, y_train, X_test, y_test, c):
    clf = LogisticRegression(penalty = 'l2', max_iter = 1000000, C = c)
    clf.fit(X_train, y_train)

    train_accuracy = clf.score(X_train, y_train)
    y_test_prediction = clf.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_test, y_test_prediction)

    return train_accuracy, test_accuracy


# Run logistic regression.
def lr_standard():
    X_train, y_train, X_test, y_test = init()
    c_values = generate_c_values(0.000001, 10)

    train_accuracies = []
    test_accuracies = []

    for c in c_values:
        train_accuracy, test_accuracy = logistic_regression(X_train, y_train, X_test, y_test, c)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    generate_plot(train_accuracies, test_accuracies, c_values)


# Implementation of logistic regression with k-fold cross validation.
def lr_kfold():
    X_train, y_train, X_test, y_test = init()
    
    c_values = generate_c_values(0.000001, 10)
    k_values = [5, 6, 7, 8, 9, 10]

    test_accuracies = []

    for k in k_values:
        print("k: " + str(k))
        fold_accuracies = []
        
        X_length = len(X_train)
        y_length = len(y_train)
        fold = X_length / k

        for c in c_values:
            print("\tc: " + str(c))
            c_accuracies = []

            for group in range(k):
                next_group = group + 1
                start = int(fold * group)
                stop = int(fold * next_group)

                if group == (k - 1):
                    X_train_group = X_train[0 : start]
                    y_train_group = y_train[0 : start]
                
                elif group == 0:
                    X_train_group = X_train[stop + 1 : X_length]
                    y_train_group = y_train[stop + 1 : y_length]

                else:
                    X_train_group = np.concatenate((
                        np.array(X_train[0 : start]),
                        np.array(X_train[stop + 1 : X_length])))
                    
                    y_train_group = np.concatenate((
                        np.array(y_train[0 : start]),
                        np.array(y_train[stop + 1 : y_length])))
                
                X_train_test = X_train[start : stop]
                y_train_test = y_train[start : stop]

                train_accuracy, test_accuracy = logistic_regression(X_train_group, y_train_group, X_train_test, y_train_test, c)
                c_accuracies.append(test_accuracy)
            
            avg_accuracy = np.average(c_accuracies)
            fold_accuracies.append(avg_accuracy)

        test_accuracies.append(fold_accuracies)

    generate_kfold_plot(test_accuracies, c_values)


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

    plt.plot(c_values, train_accuracies, 'b', label = "Train Accuracy")
    plt.plot(c_values, test_accuracies, 'g', label = "Test Accuracy")

    plt.legend()

    filename = "LogisticRegression.png"
    plt.savefig(filename)
    plt.clf()


# Use matplotlib to create graph for results (with k-fold cross validation).
def generate_kfold_plot(test_accuracies, c_values):
    plt.title("Logistic Regression with K-Fold Cross Validation")
    plt.xlabel("Regularization Parameter")
    plt.xscale('log')
    plt.ylabel("Accuracy")

    plt.plot(c_values, test_accuracies[0], 'b', label = "k = 5")
    plt.plot(c_values, test_accuracies[1], 'g', label = "k = 6")
    plt.plot(c_values, test_accuracies[2], 'r', label = "k = 7")
    plt.plot(c_values, test_accuracies[3], 'c', label = "k = 8")
    plt.plot(c_values, test_accuracies[4], 'm', label = "k = 9")
    plt.plot(c_values, test_accuracies[5], 'y', label = "k = 10")

    plt.legend()

    filename = "LogisticRegression_KFold.png"
    plt.savefig(filename)
    plt.clf()


def main():
    lr_standard()
    lr_kfold()


main()