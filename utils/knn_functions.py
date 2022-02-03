import numpy as np
import utils.corruptions as cor
from sklearn.neighbors import KNeighborsClassifier


def run_knn(train_X, train_y, test_X, test_y, neighbor_count):
    clf = KNeighborsClassifier(n_neighbors=neighbor_count).fit(train_X, train_y)

    acc = clf.score(test_X, test_y)

    return acc


def run_label_corruption_knn(train_X, train_y, test_X, test_y, num_labels, neighbor_count, corruption_levels):
    X = []
    for x in train_X:
        X.append(x.flatten())
    train_X = np.array(X)
    del X

    X = []
    for x in test_X:
        X.append(x.flatten())
    test_X = np.array(X)
    del X

    accs = {}
    for n in neighbor_count:
        accs[n] = []

    for c in corruption_levels:

        corrupted_y = train_y.copy()
        corrupted_y = cor.label_randomizer(corrupted_y, c)

        for n in neighbor_count:
            acc = round(run_knn(train_X, corrupted_y, test_X, test_y, n), 5)

            accs[n].append(acc)

    return accs