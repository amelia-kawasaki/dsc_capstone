import json
from sklearn import datasets
from sklearn.model_selection import train_test_split
from feats import unmulticlass
import numpy as np
import utils
import sklearn
from keras.datasets import mnist

def read_params(file):

    # read in parameters
    try:
        with open(file) as json_file:
            params = json.load(json_file)['params']
    except Exception:
        print('failed to read param file')
        raise

    # check config file formatting
    utils.check_config_formatting(params)

    # get model params
    alpha = None
    classification = params['data']['classification']

    if params['model']['model_type'] == 'Laplacian':
        utils.check_laplacian_config_formatting(params)
        alpha = 1
    elif params['model']['model_type'] == 'Gaussian':
        utils.check_gaussian_config_formatting(params)
        alpha = 1 / 2
    else:
        raise Exception('Given model_type unsupported')

    return params, alpha, classification


def get_raw_data(params):
    # get data
    classification = None
    if params['data']['dataset'] == 'mnist':
        #importing the data
        (x_train, y_train), (X, y) = mnist.load_data()
        # X = np.vstack((x_train, x_test))
        X = X.reshape(X.shape[0], (X.shape[1] * X.shape[2]))
        # y = np.hstack((y_train, y_test))

        y_stack = np.zeros_like(y.reshape(-1, 1))
        for i in np.unique(y):
            y_stack = np.hstack((y_stack, unmulticlass(y, i).reshape(-1, 1)))

    elif params['data']['dataset'] == 'blobs':
        X, y = datasets.make_blobs(n_samples=5_000, centers=10, random_state=0, n_features=64)

        y_stack = np.zeros_like(y.reshape(-1, 1))
        for i in np.unique(y):
            y_stack = np.hstack((y_stack, unmulticlass(y, i).reshape(-1, 1)))
    else:
        raise Exception('given dataset not supported')

    return X, y, y_stack[:, 1:]


def train_test_val_distances(X, y, y_stack, params, stage, shuffled, corruption):
    # splitting the data into train/test/validation .60, .20, .20)
    X_train, X_test, y_train, y_test, y_stack_train, y_stack_test = train_test_split(X, y, y_stack, test_size=0.25)
    X_train, X_val, y_train, y_val, y_stack_train, y_stack_val = train_test_split(X_train, y_train, y_stack_train, test_size=0.25)

    # getting the distance matrix for training
    if params['data']['distance_precalculations'] is True:
        d_train, d_test, d_val = utils.get_distance_matrices(params['data']['distance_files'], stage=stage, stored=True)
    else:
        d_train, d_test, d_val = utils.get_distance_matrices(params['data']['distance_files'], stage=stage, mats=[X_train, X_test, X_val])

    print('distance matrices loaded')
    return [(X_train, d_train, y_train, y_stack_train), (X_test, d_test, y_test, y_stack_test), (X_val, d_val, y_val, y_stack_val)]


def etl_data(params, shuffled=0.0, corruption=0.0, stage=None):
    X, y, y_stack = get_raw_data(params)
    if stage == 'test':
        X = X[:10, :]
        y = y[:10]
        y_stack = y_stack[:10, :]
    data = train_test_val_distances(X, y, y_stack, params, stage, shuffled, corruption)
    return data



