#####
## Noah Serr V00891494
## SENG 474 A2
## Support Vector Machines
#####

import numpy as np
from sklearn import metrics
from sklearn import svm
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
            sandal_count += 1

        if y_train[i] == 7 and sneaker_count < 1000:
            y_train_temp.append(1)
            X_train_temp.append(X_train[i])
            sneaker_count += 1

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


# Implementation of support vector machines.
def svm_function(X_train, y_train, X_test, y_test, kernel, c, gamma):
    if kernel == 'linear': clf = svm.SVC(kernel = kernel, C = c)
    elif kernel == 'rbf': clf = svm.SVC(kernel = kernel, C = c, gamma = gamma)
    
    clf.fit(X_train, y_train)

    train_accuracy = clf.score(X_train, y_train)
    y_test_prediction = clf.predict(X_test)
    test_accuracy = metrics.accuracy_score(y_test, y_test_prediction)

    return train_accuracy, test_accuracy


# Run SVM.
def svm_standard(kernel, gamma):
    X_train, y_train, X_test, y_test = init()
    c_values = generate_c_values(0.000001, 10)

    train_accuracies = []
    test_accuracies = []

    for c in c_values:
        train_accuracy, test_accuracy = svm_function(X_train, y_train, X_test, y_test, kernel, c, gamma)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    generate_plot(train_accuracies, test_accuracies, c_values, kernel)


# Implementation of support vector machines with k-fold cross validation.
def svm_kfold(kernel, gamma):
    X_train, y_train, X_test, y_test = init()
    
    c_values = generate_c_values(0.000001, 10)
    k_values = [10]

    test_accuracies = []

    for k in k_values:
        fold_accuracies = []
        
        X_length = len(X_train)
        y_length = len(y_train)
        fold = X_length / k

        for c in c_values:
            print("c: " + str(c))
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

                train_accuracy, test_accuracy = svm_function(X_train_group, y_train_group, X_train_test, y_train_test, kernel, c, gamma)
                c_accuracies.append(test_accuracy)
            
            avg_accuracy = np.average(c_accuracies)
            fold_accuracies.append(avg_accuracy)

        test_accuracies.append(fold_accuracies)

    generate_kfold_plot(test_accuracies, c_values, kernel)


# Implementation of support vector machines with k-fold cross validation and a Gaussian kernel.
def svm_gaussian():
    X_train, y_train, X_test, y_test = init()

    c_values = [0.1]
    k_values = [10]
    gamma_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

    test_accuracies = []
    training_accuracies = []

    for k in k_values:
        X_length = len(X_train)
        y_length = len(y_train)
        fold = X_length / k

        for c in c_values:
            print("c: " + str(c))
            c_test_accuracies = []
            c_training_accuracies = []

            for gamma in gamma_values:
                print("\tgamma: " + str(gamma))
                gamma_test_accuracies = []
                gamma_training_accuracies = []

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

                    train_accuracy, test_accuracy = svm_function(X_train_group, y_train_group, X_train_test, y_train_test, 'rbf', c, gamma)
                    gamma_test_accuracies.append(test_accuracy)
                    gamma_training_accuracies.append(train_accuracy)

                avg_test_accuracy = np.average(gamma_test_accuracies)
                c_test_accuracies.append(avg_test_accuracy)

                avg_train_accuracy = np.average(gamma_training_accuracies)
                c_training_accuracies.append(avg_train_accuracy)

            test_accuracies.append(c_test_accuracies)
            training_accuracies.append(c_training_accuracies)

        generate_gaussian_plot(training_accuracies, test_accuracies, gamma_values)


# Generates regulization parameters using a logarithmically spaced grid.
def generate_c_values(initial, alpha):
    c_values = []

    for i in range(0, 10):
        c_values.append(initial * (alpha ** i))
    
    return c_values


# Use matplotlib to create graph for results.
def generate_plot(train_accuracies, test_accuracies, c_values, kernel):
    plt.title("Support Vector Machines")
    plt.xlabel("Regularization Parameter")
    plt.xscale('log')
    plt.ylabel("Accuracy")

    plt.plot(c_values, train_accuracies, 'b', label = "Train Accuracy")
    plt.plot(c_values, test_accuracies, 'g', label = "Test Accuracy")

    plt.legend()

    if kernel == "linear": filename = "LinearSVM.png"
    elif kernel == "rbf": filename = "GaussianSVM.png"

    plt.savefig(filename)
    plt.clf()


# Use matplotlib to create graph for results (with k-fold cross validation).
def generate_kfold_plot(test_accuracies, c_values, kernel):
    plt.title("Support Vector Machines with K-Fold Cross Validation")
    plt.xlabel("Regularization Parameter")
    plt.xscale('log')
    plt.ylabel("Accuracy")

    plt.plot(c_values, test_accuracies[0], 'b', label = "k = 10")

    plt.legend()

    if kernel == "linear": filename = "LinearSVM_KFold.png"
    elif kernel == "rbf": filename = "GaussianSVM_KFold.png"

    plt.savefig(filename)
    plt.clf()


# Use matplotlib to create graph for results (Gaussian kernel).
def generate_gaussian_plot(training_accuracies, test_accuracies, gamma_values):
    plt.title("Support Vector Machines with K-Fold Cross Validation\n and Gaussian Kernel (C = 0.1, K = 10)")
    plt.xlabel("Gamma")
    plt.xscale('log')
    plt.ylabel("Accuracy")

    plt.plot(gamma_values, test_accuracies[0], 'b', label = "Test Accuracy")
    plt.plot(gamma_values, training_accuracies[0], 'g', label = "Training Accuracy")

    plt.legend()

    filename = "GaussianSVM_KFold.png"
    plt.savefig(filename)
    plt.clf()


def main():
    # svm_standard('linear', 'scale')
    # svm_kfold('linear', 'scale')
    svm_gaussian()


main()