""" This module implement least square classification.

@authors: Valentin
"""
import pandas as pd
import numpy as np
from generate_data import generate_demo_data


def augmented_input(data):
    """ Reformate X input data into augmented (1, X) vector. """
    return pd.DataFrame([1] * len(data)).join(data).as_matrix().astype(float)


def pseudo_inv(X):
    """ Compute brutally pseudo inverse matrix. """
    tmp = np.dot(X.transpose(), X)
    tmp = np.linalg.inv(tmp)
    tmp = np.dot(tmp, X.transpose())
    return tmp


def decision_bundaries(W, xmin=-5, xmax=5):
    """ Compute decision bundary. """
    xstep = np.arange(xmin, xmax, 0.1)
    y = []
    c1 = W[:, 0]
    c2 = W[:, 1]
    c3 = W[:, 2]
    d1 = c2 - c1
    d2 = c2 - c3
    for x in x
    y.append((-c[0] - c[1] * x) / c[2])
    return x, y


def least_square_classification(df):
    """ Compute least_square_classification. """
    X = augmented_input(df.iloc[:, :2])
    Xpi = pseudo_inv(X)
    W = np.dot(Xpi, df.iloc[:, 2:].as_matrix().squeeze())
    return W


def plot_data_and_decision_bundaries(df, x, y):
    """ cf name. """
    import matplotlib.pyplot as plt
    plt.scatter(df[df['class0'] == 1]['x'],
                df[df['class0'] == 1]['y'],
                color='r')
    plt.scatter(df[df['class1'] == 1]['x'],
                df[df['class1'] == 1]['y'],
                color='b')
    plt.scatter(df[df['class2'] == 1]['x'],
                df[df['class2'] == 1]['y'],
                color='g')
    for b in y:
        plt.plot(x, b, color='k')
    plt.show()


if __name__ == '__main__':
    df = generate_demo_data(500)
    X = augmented_input(df.iloc[:, :2])
    W = least_square_classification(df)
    x, y = decision_bundaries(W)
    plot_data_and_decision_bundaries(df, x, y)
