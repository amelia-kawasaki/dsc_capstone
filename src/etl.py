import json
from sklearn import datasets
from sklearn.model_selection import train_test_split
from feats import unmulticlass
import utils


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
        # importing the data
        dataset = datasets.load_digits()
        X = dataset.data
        y = dataset.target

        if params['data']['multiclass'] is False:
            X, y = unmulticlass(X, y, (0, 1))
        else:
            raise Exception('multiclass not supported for this dataset yet')
            # TODO support multiclass
    elif params['data']['dataset'] == 'diabetes':
        dataset = datasets.load_diabetes()
        X = dataset.data
        y = dataset.target

    elif params['data']['dataset'] == 'random regression':
        X, y = utils.load_random_regression()
    else:
        raise Exception('given dataset not supported')

    return X, y


def train_test_val_distances(X, y, params, stage):
    # splitting the data into train/test/validation .60, .20, .20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


    # getting the distance matrix for training
    if params['data']['distance_precalculations'] is True:
        d_train, d_test, d_val = utils.get_distance_matrices(params['data']['distance_files'], stage=stage, stored=True)
    else:
        d_train, d_test, d_val = utils.get_distance_matrices(params['data']['distance_files'], stage=stage, mats=[X_train, X_test, X_val])

    return [(X_train, d_train, y_train), (X_test, d_test, y_test), (X_val, d_val, y_val)]


def etl_data(config_file, stage=None):
    params, alpha, classification = read_params(config_file)
    X, y = get_raw_data(params)
    data = train_test_val_distances(X, y, params, stage=stage)
    return data, params, alpha, classification