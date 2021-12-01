"""
Checkpoint Code - Amelia Kawasaki
Fall 2021, Capstone Project with Professor Belkin
"""
from sklearn import datasets
import numpy as np
from scipy.spatial import distance_matrix
from numpy.linalg import inv as invert
from sklearn.metrics import mean_squared_error, classification_report, r2_score
import os
from matplotlib import pyplot as plt


class LaplacianKernel:
    def __init__(self, alpha=1, n=1, t=None):
        self.kernel_matrix = None
        self.alpha = None
        self.training_data = None
        self.n = n
        self.alpha1 = alpha
        self.t = t

    def set_t(self, t):
        self.t = t

    def get_t(self):
        return self.t

    def create_matrix(self, d_matrix):
        """
        Compute the laplacian kernel between X and Y.
        The laplacian kernel is defined as::
            K(x, y) = exp(||x-y||^(beta/n) * (-1/t)^(1/n))
        for each pair of rows x in X and y in Y.
        :param d_matrix: nxn np array
        :return: K: nxn np array
        """
        if self.t is None:
            length = (d_matrix.shape[0] ** 2) - d_matrix.shape[0]
            self.t = np.sum(d_matrix) / (2 * (length))
        numerator = d_matrix ** (self.alpha1/self.n)
        numerator = numerator * (-1)
        denominator = self.t ** (1/self.n)
        K = np.exp(numerator / denominator)
        return K

    def fit(self, X, y, distances=None):
        """
        Fits data and target to a laplacian kernel model
        :param X: nxm np array
        :param y: nx1 np array
        :param distances: nxn np array
        :return:
        """
        if distances is not None:
            d_matrix = distances
        else:
            d_matrix = distance_matrix(X, X)

        K = self.create_matrix(d_matrix)
        self.kernel_matrix = K

        self.training_data = X
        lstsq = np.linalg.lstsq(K, y, rcond=None)
        self.alpha = lstsq[0]

    def predict(self, X, distances=None, classification=True):
        """
        Predicts labels y given new data X
        :param X: pxm np array
        :param distances: pxp np array
        :param classification: boolean
        :return:
        """

        if self.alpha is None:
            raise ValueError("Model has not been fit")

        elif self.kernel_matrix is None:
            raise ValueError("Model has not been fit")

        else:
            if distances is not None:
                d_matrix = distances
            else:
                d_matrix = distance_matrix(self.training_data, X)

            K = self.create_matrix(d_matrix)
            y = np.dot(self.alpha, K)

            return y


def load_random_regression():
    n_samples = 1000
    n_outliers = 50

    X, y, coef = datasets.make_regression(
        n_samples=n_samples,
        n_features=1,
        n_informative=1,
        noise=100,
        coef=True,
        random_state=0,
    )

    # Add outlier data
    # np.random.seed(0)
    # X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
    # y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

    plt.scatter(X, y)
    plt.show()
    return X, y


def get_distance_matrices(title, stage=None, stored=False, mats=None):
    """
    Creates or loads matrices of distances between data points
    :param title: string
    :param stored: boolean
    :param mats: list of length 3
    :return: np array, np array, np array
    """
    dir = os.path.dirname(os.path.realpath(__file__))
    if stage == 'test':
        dir2 = os.path.join(dir, '..', 'test', 'testdata')
    else:
        dir2 = os.path.join(dir, '..', 'data')
    m_list = []

    if stored is True:
        for i in ['train', 'test', 'val']:
            path = os.path.join(dir2, title + '_' + i + '.npy')
            m_list.append(np.load(path))

    else:
        d_train = distance_matrix(mats[0], mats[0])
        path = os.path.join(dir2, title + '_train')
        np.save(path, d_train)
        m_list.append(d_train)
        d_test = distance_matrix(mats[0], mats[1])
        path = os.path.join(dir2, title + '_test')
        np.save(path, d_test)
        m_list.append(d_test)
        d_val = distance_matrix(mats[0], mats[2])
        path = os.path.join(dir2, title + '_val')
        np.save(path, d_val)
        m_list.append(d_val)

    return m_list[0], m_list[1], m_list[2]


def check_config_formatting(params):
    for i in ['data', 'model', 'validation', 'testing']:
        if i not in params.keys():
            raise ValueError('config file not formatted correctly')
    for i in ['dataset', 'classification', 'distance_precalculations', 'distance_files']:
        if i not in params['data'].keys():
            raise ValueError('"data" in config file is not formatted correctly')
    for i in ['model_type']:
        if i not in params['model'].keys():
            raise ValueError('"model" in config file is not formatted correctly')
    for i in ['validation_param']:
        if i not in params['validation'].keys():
            raise ValueError('"validation" in config file is not formatted correctly')


def check_laplacian_config_formatting(params):
    # for i in ['t']:
    #     if i not in params['model'].keys():
    #         raise ValueError('correct parameters not specified for model training (Laplacian)')
    return


def check_gaussian_config_formatting(params):
    # for i in ['t']:
    #     if i not in params['model'].keys():
    #         raise ValueError('correct parameters not specified for model training (Laplacian)')
    return


def comparison_metric(a, b, classification):
    if classification:
        r = sum(a == b)/ len(a)
    else:
        r = mean_squared_error(a, b)
    return r


def validation(m, m_params, X_train, y_train, d_train, X_val, y_val, d_val, classification):
    # validation for laplacian kernel model
    # grid search for an optimal value of t
    print('----------Training-------------')
    model = m(alpha=m_params[0])
    model.fit(X_train, y_train, distances=d_train)
    prediction = model.predict(X_train, distances=d_train, classification=classification)
    # compare = comparison_metric(prediction_r, y_train, classification)

    scores = []
    grid_search = [None, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    alpha = m_params[0]

    for i in grid_search:
        model = m(t=i, alpha=alpha)
        model.fit(X_train, y_train, distances=d_train)
        prediction = model.predict(X_val, distances=d_val, classification=classification)
        compare = comparison_metric(prediction, y_val, classification)
        scores.append(compare)

    # print('----------Validation-----------')
    idx = np.argmax(scores)
    i = grid_search[idx]
    model = m(t=i, alpha=alpha)
    model.fit(X_train, y_train, distances=d_train)
    prediction = model.predict(X_train, distances=d_train, classification=classification)
    return model, prediction


def testing(model, X_test, y_test, d_test, classification):

    prediction = model.predict(X_test, distances=d_test, classification=classification)
    return prediction

