"""
Checkpoint Code - Amelia Kawasaki
Fall 2021, Capstone Project with Professor Belkin
"""

import etl
import utils
from argparse import ArgumentParser
import os

parser = ArgumentParser()
parser.add_argument('target', type=str, metavar="TARGET")
parser.add_argument('-f', '--file', type=str, metavar="f", help='configuration file for targets: all')
args = parser.parse_args()

if args.target == 'test':
    print("testing with test data")
    path = os.path.join("..", "test", "test_config.json")
    data, params, alpha, classification = etl.etl_data(path)
elif args.target == 'all':
    print("running with " + str(args.file))
    data, params, alpha, classification = etl.etl_data(args.file)
else:
    raise ValueError('Invalid Arguments')

[(X_train, d_train, y_train), (X_test, d_test, y_test), (X_val, d_val, y_val)] = data

# creating kernel model, validating for t
model = utils.LaplacianKernel
model = utils.validation(model, [alpha], X_train, y_train, d_train, X_val, y_val, d_val, classification)
model = utils.testing(model, X_test, y_test, d_test, classification)
