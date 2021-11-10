

def unmulticlass(X, y, labels):
    # only taking two classes and re-labeling to -1 and 1
    y_idx = (y == labels[0]) | (y == labels[1])
    y = y[y_idx]
    X = X[y_idx]
    y[y == 0] = -1
    return X, y