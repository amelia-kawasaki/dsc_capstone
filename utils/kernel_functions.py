import numpy as np
import utils.corruptions as cor

def euclidean_distance(X, Y):
    D = np.zeros((X.shape[0], Y.shape[0]))
    
    if Y is X:
        for i in range(X.shape[0] - 1):
            distances = np.sqrt(np.sum((X[(i + 1):] - X[i])**2, axis = 1))
            D[i, (i + 1):] = distances
            D[(i + 1):, i] = distances
            
    else:
        for i in range(X.shape[0]):
            D[i] = np.sqrt(np.sum((Y - X[i])**2, axis = 1))
            
    return D

def minkowski_distance(X, Y, p = 2):
    D = np.zeros((X.shape[0], Y.shape[0]))
    
    if Y is X:
        for i in range(X.shape[0] - 1):
            distances = (np.sum((abs(X[(i + 1):] - X[i]))**p, axis = 1))**(1/p)
            D[i, (i + 1):] = distances
            D[(i + 1):, i] = distances
            
    else:
        for i in range(X.shape[0]):
            D[i] = (np.sum((abs(Y - X[i]))**p, axis = 1))**(1/p)
            
    return D

def kernel_func_p(x, sigma, p = 1):
    return np.exp(-((x**p)/(p * sigma**p)))

def solve_alpha(K, y):
    return np.matmul(np.linalg.inv(K), y)

def create_y_matrix(y_train, num_digits):
    y_mat = np.zeros((len(y_train), num_digits))
    for i in range(len(y_train)):
        y_mat[i][y_train[i]] = 1
    return y_mat

def classification(predictions, y_test):
    preds = np.argmax(predictions, axis = 1)
    correct = 0
    for i, j in zip(preds, y_test):
        if i == j:
            correct += 1
    return correct / len(y_test)

def run_kernel(train_X, train_y, test_X, test_y, num_labels, power, D_train = None, D_test = None):
    """
    D_train and D_test can be set to avoid recalculating
    the same distance matrices for different power kernels
    """
    
    if D_train is None:
        D_train = euclidean_distance(train_X, train_X)
        
    if D_test is None:
        D_test = euclidean_distance(test_X, train_X)

    K_train = kernel_func_p(D_train, np.mean(D_train), power)
    K_train_inv = np.linalg.inv(K_train)
    K_test = kernel_func_p(D_test, np.mean(D_train), power)

    y_matrix = create_y_matrix(train_y, num_labels)

    alpha = np.matmul(K_train_inv, y_matrix)

    predictions = np.matmul(K_test, alpha)
    kernel_acc = round(classification(predictions, test_y), 5)

    return kernel_acc

def run_label_corruption_kernels(train_X, train_y, test_X, test_y, num_labels, powers, corruption_levels):
    
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

    D_train = euclidean_distance(train_X, train_X)

    D_test = euclidean_distance(test_X, train_X)

    accs = {}
    for p in powers:
        accs[p] = []

    for c in corruption_levels:
        corrupted_y = train_y.copy()
        corrupted_y = cor.label_randomizer(corrupted_y, c)

        for p in powers:
            acc = run_kernel(None, corrupted_y, test_X, test_y, num_labels, p, D_train, D_test)
            
            accs[p].append(acc)

    return accs

def run_random_corruption_kernels(train_X, train_y, test_X, test_y, num_labels, powers, corruption_levels):

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
    for p in powers:
        accs[p] = []

    for c in corruption_levels:

        corrupted_X = train_X.copy()
        corrupted_X = cor.random_filter(corrupted_X, c)

        D_train = euclidean_distance(corrupted_X, corrupted_X)
        D_test = euclidean_distance(test_X, corrupted_X)

        for p in powers:
            acc = run_kernel(None, train_y, test_X, test_y, num_labels, p, D_train, D_test)
            
            accs[p].append(acc)

    return accs

def run_gaussian_blur_kernels(train_X, train_y, test_X, test_y, num_labels, powers, filter_sizes, sigmas):

    X = []
    for x in test_X:
        X.append(x.flatten())
    test_X = np.array(X)
    del X
    
    accs = {}
    for f in filter_sizes:
        accs[f] = {}
        for p in powers:
            accs[f][p] = []

    for f in filter_sizes:
        for s in sigmas:

            corrupted_X = cor.gaussian_filter(train_X, f, s)

            D_train = euclidean_distance(corrupted_X, corrupted_X)
            D_test = euclidean_distance(test_X, corrupted_X)

            for p in powers:
                acc = run_kernel(None, train_y, test_X, test_y, num_labels, p, D_train, D_test)
                
                accs[f][p].append(acc)
    
    return accs