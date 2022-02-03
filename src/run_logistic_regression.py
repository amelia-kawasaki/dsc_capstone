"""
Checkpoint Code - Amelia Kawasaki
Fall 2021, Capstone Project with Professor Belkin
"""
import numpy as np
import etl
import utils
from argparse import ArgumentParser
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('target', type=str, metavar="TARGET")
parser.add_argument('-f', '--file', type=str, metavar="f", help='configuration file for targets: all')
args = parser.parse_args()

if args.target == 'test':
    print("testing with test data")
    dir = os.path.dirname(os.path.realpath(__file__))
    path = os.path.join(dir, "..", "test", "test_config.json")
    print(path)
elif args.target == 'all':
    print("running with " + str(args.file))
    path = args.file
else:
    raise ValueError('Invalid Arguments')

params, alpha, classification = etl.read_params(path)


if args.target == 'test':
    data = etl.etl_data(params, shuffled=0.0, stage=args.target)

    [(X_train, d_train, y_train, y_stack_train), (X_test, d_test, y_test, y_stack_test), (X_val, d_val, y_val, y_stack_val)] = data


elif args.target == 'all':

    X_results = []
    y_results = []

    for j in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        data = etl.etl_data(params, shuffled=0.0, corruption=j, stage=args.target)

        [(X_train, d_train, y_train, y_stack_train), (X_test, d_test, y_test, y_stack_test),
         (X_val, d_val, y_val, y_stack_val)] = data

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        report = classification_report(y_test, pred, output_dict=True)
        X_results.append(j)
        y_results.append(report['accuracy'])

    plt.plot(X_results, y_results)
    plt.title('Logistic Regression: Corruption vs Accuracy')
    plt.xlabel('Corruption in Training Set')
    plt.ylabel('Accuracy')
    plt.show()

