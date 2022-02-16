import numpy as np
import utils.corruptions as cor
from sklearn.ensemble import RandomForestClassifier

def run_forest(train_X, train_y, test_X, test_y, num_trees):
    
    clf = RandomForestClassifier(n_estimators = num_trees, criterion = 'entropy').fit(train_X, train_y)
    
    acc = clf.score(test_X, test_y)
    
    return acc

def run_label_corruption_forests(train_X, train_y, test_X, test_y, num_labels, forest_sizes, corruption_levels):
    
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
    for n in forest_sizes:
        accs[n] = []
    
    for c in corruption_levels:
        
        corrupted_y = train_y.copy()
        corrupted_y = cor.label_randomizer(corrupted_y, c)
        
        for n in forest_sizes:
            
            acc = round(run_forest(train_X, corrupted_y, test_X, test_y, n), 5)
            
            accs[n].append(acc)
        
    return accs

def run_random_corruption_forests(train_X, train_y, test_X, test_y, num_labels, forest_sizes, corruption_levels):
    
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
    for n in forest_sizes:
        accs[n] = []
        
    min_val = np.min(train_X)
    max_val = np.max(train_X)
    
    for c in corruption_levels:
        
        corrupted_X = train_X.copy()
        corrupted_X = cor.random_filter(corrupted_X, c, min_val, max_val)
        
        for n in forest_sizes:
            
            acc = round(run_forest(corrupted_X, train_y, test_X, test_y, n), 5)
            
            accs[n].append(acc)
        
    return accs

def run_gaussian_blur_forests(train_X, train_y, test_X, test_y, num_labels, forest_sizes, filter_sizes, sigmas):
    
    X = []
    for x in test_X:
        X.append(x.flatten())
    test_X = np.array(X)
    del X
    
    accs = {}
    for f in filter_sizes:
        accs[f] = {}
        for n in forest_sizes:
            accs[f][n] = []
            
    for f in filter_sizes:
        for s in sigmas:
            
            corrupted_X = cor.gaussian_filter(train_X, f, s)
            
            for n in forest_sizes:
                
                acc = round(run_forest(corrupted_X, train_y, test_X, test_y, n), 5)
                
                accs[f][n].append(acc)
    
    return accs

def run_multi_forests(train_X, train_y, test_X, test_y, num_labels, forest_sizes, corruption_levels, cor_type = 'label', num_iter = 3):
    all_accs = {}
    for n in forest_sizes:
        all_accs[n] = []
    
    for _ in range(num_iter):
        if cor_type == 'label':
            iter_acc = run_label_corruption_forests(train_X, train_y, test_X, test_y, num_labels, forest_sizes, corruption_levels)
        elif cor_type == 'random':
            iter_acc = run_random_corruption_forests(train_X, train_y, test_X, test_y, num_labels, forest_sizes, corruption_levels)
            
        for n in forest_sizes:
            all_accs[n].append(iter_acc[n])
            
    return all_accs
