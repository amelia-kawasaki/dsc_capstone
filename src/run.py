"""
Checkpoint Code - Amelia Kawasaki
Fall 2021, Capstone Project with Professor Belkin
"""
import numpy as np

import etl
import utils
from argparse import ArgumentParser
import os
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

    # creating kernel model, validating for t
    y_final_test = np.zeros((X_test.shape[0], y_stack_train.shape[1]))
    y_final_train = np.zeros((X_train.shape[0], y_stack_train.shape[1]))
    for i in range(y_stack_train.shape[1]):
        model = utils.LaplacianKernel
        model, train_prediction = utils.validation(model, [alpha], X_train, y_stack_train[:, i], d_train, X_val,
                                                   y_stack_val[:, i], d_val, classification)
        prediction = utils.testing(model, X_test, y_stack_test[:, i], d_test, classification)
        y_final_test[:, i] = prediction
        y_final_train[:, i] = train_prediction

    y1 = y_final_test.argmax(axis=1)
    y2 = y_final_train.argmax(axis=1)
    print('Finished')


elif args.target == 'all':

    X_results = []
    y_results = []

    data = etl.etl_data(params, stage=args.target)

    [(X_train, d_train, y_train, y_stack_train), (X_test, d_test, y_test, y_stack_test),
     (X_val, d_val, y_val, y_stack_val)] = data

    for j in [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

        y_train, y_stack_train = utils.shuffle_labels(y_train, y_stack_train, j)

        # creating kernel model, validating for t
        y_final_test = np.zeros((X_test.shape[0], y_stack_train.shape[1]))
        y_final_train = np.zeros((X_train.shape[0], y_stack_train.shape[1]))
        for i in range(y_stack_train.shape[1]):
            model = utils.LaplacianKernel
            model, train_prediction = utils.validation(model, [alpha], X_train, y_stack_train[:, i], d_train, X_val, y_stack_val[:, i], d_val, classification)
            prediction = utils.testing(model, X_test, y_stack_test[:, i], d_test, classification)
            y_final_test[:, i] = prediction
            y_final_train[:, i] = train_prediction


        y1 = y_final_test.argmax(axis=1)
        y2 = y_final_train.argmax(axis=1)
        # print('Training Results')
        # print(classification_report(y_train, y2))
        # print('Testing Results')
        report = classification_report(y_test, y1, output_dict=True)
        # print(report)
        X_results.append(j)
        y_results.append(report['accuracy'])
        print('finished ' + str(j))

    plt.plot(X_results, y_results)
    plt.title('Gaussian Kernel: Label Corruption vs Accuracy')
    plt.xlabel('Proportion of Shuffled Labels in Training Set')
    plt.ylabel('Accuracy')
    plt.show()