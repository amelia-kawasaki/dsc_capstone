import numpy as np
import utils.corruptions as cor
from sklearn.ensemble import RandomForestClassifier

def run_forest(train_X, train_y, test_X, test_y, num_trees):
    
    clf = RandomForestClassifier(n_estimators = num_trees, criterion = 'entropy').fit(train_X, train_y)
    
    error = 1 - clf.score(test_X, test_y)
    
    return error

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
    
    errors = {}
    for n in forest_sizes:
        errors[n] = []
    
    for c in corruption_levels:
        
        corrupted_y = train_y.copy()
        corrupted_y = cor.label_randomizer(corrupted_y, c)
        
        for n in forest_sizes:
            
            error = round(run_forest(train_X, corrupted_y, test_X, test_y, n), 5)
            
            errors[n].append(error)
        
    return errors

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
    
    errors = {}
    for n in forest_sizes:
        errors[n] = []
    
    for c in corruption_levels:
        
        corrupted_X = train_X.copy()
        corrupted_X = cor.random_filter(corrupted_X, c)
        
        for n in forest_sizes:
            
            error = round(run_forest(corrupted_X, train_y, test_X, test_y, n), 5)
            
            errors[n].append(error)
        
    return errors

def run_gaussian_blur_forests(train_X, train_y, test_X, test_y, num_labels, forest_sizes, filter_sizes, sigmas):
    
    X = []
    for x in test_X:
        X.append(x.flatten())
    test_X = np.array(X)
    del X
    
    errors = {}
    for f in filter_sizes:
        errors[f] = {}
        for n in forest_sizes:
            errors[f][n] = []
            
    for f in filter_sizes:
        for s in sigmas:
            
            corrupted_X = cor.gaussian_filter(train_X, f, s)
            
            for n in forest_sizes:
                
                error = round(run_forest(corrupted_X, train_y, test_X, test_y, n), 5)
                
                errors[f][n].append(error)
    
    return errors