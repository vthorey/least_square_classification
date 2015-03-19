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
    xstep = np.arange(xmin, xmax, 0.01)
    points = []
    c1 = W[:, 0]
    c2 = W[:, 1]
    c3 = W[:, 2]
    d1 = c2 - c1
    d2 = c2 - c3
    for x in xstep:
        tmp1 = (-d1[0] - d1[1] * x) / d1[2]
        tmp2 = (-d2[0] - d2[1] * x) / d2[2]
        if d2[0] + d2[1] * x + d2[2] * tmp1 > 0:
            points.append([x, tmp1])
        if d1[0] + d1[1] * x + d1[2] * tmp2 > 0:
            points.append([x, tmp2])

    c1 = W[:, 1]
    c2 = W[:, 2]
    c3 = W[:, 0]
    d1 = c2 - c1
    d2 = c2 - c3
    for x in xstep:
        tmp1 = (-d1[0] - d1[1] * x) / d1[2]
        tmp2 = (-d2[0] - d2[1] * x) / d2[2]
        if d2[0] + d2[1] * x + d2[2] * tmp1 > 0:
            points.append([x, tmp1])
        if d1[0] + d1[1] * x + d1[2] * tmp2 > 0:
            points.append([x, tmp2])

    return np.array(points)


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
    plt.scatter(x, y, color='k')
    plt.ylim([-6, 6])
    plt.xlim([-6, 6])
    plt.show()


if __name__ == '__main__':
    df = generate_demo_data(2000)
    X = augmented_input(df.iloc[:, :2])
    W = least_square_classification(df)
    points = decision_bundaries(W)
    x = points[:, 0]
    y = points[:, 1]
    plot_data_and_decision_bundaries(df, x, y)
