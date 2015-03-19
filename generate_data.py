""" Module to generate sets of points in the plan.

@authors: Valentin
"""
import numpy as np
import pandas as pd


def generate_gaussian_data(n, mu, sigma):
    """ Generate n points according to gaussian distribution.

    :params n: number of points
    :params mu: mean of the gaussian
    :params simga: variance of the gaussian

    :type n: int
    :type mu: tuple
    :type sigma: tuple

    :return: list of tuples containing corrdinates
    :rtype: list

    :example:
    e,g out = generate_gaussian_data(2, (2, 2), (1, 1)
        out -> [(2.4, 1.8), (2.3, 2.2)]
    """
    return [tuple(np.random.normal(mu, sigma)) for i in range(n)]


def generate_K_gaussians(n, mus, sigmas):
    """ Generate K classes according to gaussian distribution.

    :param n: Number of points to generate for each class
    :param mus: Means of each class gaussian
    :params sigmas: Variance of each class gaussian
    :type n: int
    :type mus: list of tuples
    :type sigmas: list of tuples

    :return: A list of K element, each are arrays of points
    :rtype: list

    :example:
    generate_K_gaussians(20, [(2,), (3,)], [(0.1,), (0.2,)])
    """
    # Check input integrity
    try:
        lmus = len(mus[0])
        lsigmas = len(sigmas[0])
    except TypeError:
        raise TypeError("You need to input a list of mus and sigmas")

    assert lmus == lsigmas, 'Dimension of mean and variance must be the same'
    assert len(mus) == len(sigmas), 'You need as much mean as variance'

    list_point = []
    for mu, sigma in zip(mus, sigmas):
        assert len(mu) == lmus, "all mean must be the same size"
        assert len(sigma) == lsigmas, "all sigmas must be the same size"
        list_point.append(generate_gaussian_data(n, mu, sigma))

    return list_point


def convert_list_point_to_df(list_point):
    """ Convert list of points into a Dataframe.

    :params list_point: List of list of points
    :type list_point: list

    :return: Dataframe with all points and a class column corresponding
    to each list
    :rtype: pandas.DataFrame

    :exemple:
    convert_list_point_to_df([
                              [(1.2, 1.3), (1.5, 1.6)],
                              [(1,2, 1.1)]
                              ])
        x   y   class
    0   1.2 1.3 0
    1   1.5 1.6 0
    2   1.2 1.1 1
    """
    k = len(list_point)
    class_labels = ['class' + str(i) for i in range(k)]
    columns = ['x'] + ['y'] + class_labels
    result = pd.DataFrame(columns=columns)
    for i, list_class_i in enumerate(list_point):
        tmp = pd.DataFrame(list_class_i, columns=['x', 'y'])
        zeros = np.zeros([len(tmp), k])
        tmp_class = pd.DataFrame(zeros,
                                 index=tmp.index.tolist(),
                                 columns=class_labels)
        tmp_class['class' + str(i)] = 1
        tmp = tmp.join(tmp_class)
        result = pd.concat([result, tmp], ignore_index=True)
    return result


def plot_2D_K_classes(points):
    """ Plot list of 2D points with different colors by classes. """
    import seaborn as sns
    import matplotlib.pyplot as plt
    if type(points) != pd.core.frame.DataFrame:
        df = convert_list_point_to_df(points)
    else:
        df = points
    g = sns.FacetGrid(df,
                      hue="class",
                      palette="Set1",
                      size=5)
    g.map(plt.scatter, "x", "y", s=100, linewidth=.5, edgecolor="white")
    g.add_legend()
    plt.show()


def generate_demo_data(n):
    """ Generate a demo data set with 3 classes and n points per classes. """
    mus = [(np.random.randint(-3, 3), np.random.randint(-3, 3)),
           (np.random.randint(-3, 3), np.random.randint(-3, 3)),
           (np.random.randint(-3, 3), np.random.randint(-3, 3))]
    sigmas = [(np.random.random(), np.random.random()),
              (np.random.random(), np.random.random()),
              (np.random.random(), np.random.random())]
    return convert_list_point_to_df(generate_K_gaussians(n, mus, sigmas))


def generate_demo_data_2(n):
    """ Generate a demo data set with 3 classes and n points per classes. """
    mus = [(1, 2), (0, 0)]
    sigmas = [(0.5, 0.42), (0.1, 0.15)]
    return convert_list_point_to_df(generate_K_gaussians(n, mus, sigmas))
