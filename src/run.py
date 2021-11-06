"""
Checkpoint Code - Amelia Kawasaki
Fall 2021, Capstone Project with Professor Belkin
"""

from sklearn import datasets
from utils import LaplacianKernel, get_distance_matrices, check_config_formatting, check_laplacian_config_formatting, \
    check_gaussian_config_formatting, validation
from sklearn.model_selection import train_test_split
from feats import unmulticlass
import json

# read in parameters
try:
    with open('config.json') as json_file:
        params = json.load(json_file)['params']
except Exception:
    print('failed to read param file')
    raise

# check config file formatting
check_config_formatting(params)

# get data
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
else:
    raise Exception('given dataset not supported')


# splitting the data into train/test/validation .60, .20, .20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


# getting the distance matrix for training
if params['data']['distance_precalculations'] is True:
    d_train, d_test, d_val = get_distance_matrices('digits', stored=True)
else:
    d_train, d_test, d_val = get_distance_matrices('digits', mats=[X_train, X_test, X_val])


# get model params
alpha = None

if params['model']['model_type'] == 'Laplacian':
    check_laplacian_config_formatting(params)
    alpha = 1

elif params['model']['model_type'] == 'Gaussian':
    check_gaussian_config_formatting(params)
    alpha = 1/2

else:
    raise Exception('Given model_type unsupported')


# creating laplacian kernel model, validating for t
model = LaplacianKernel

validation(model, [alpha], X_train, y_train, d_train, X_val, y_val, d_val)
