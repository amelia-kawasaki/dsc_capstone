from sklearn import datasets
import numpy as np

dataset = datasets.load_digits()
X = dataset.data
y = dataset.target
classification = True

np.save('/testdata/X_digits_test', X[:10, :])
np.save('/testdata/y_digits_test', y[:10])