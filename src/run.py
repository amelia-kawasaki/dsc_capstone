"""
Checkpoint Code - Amelia Kawasaki
Fall 2021, Capstone Project with Professor Belkin
"""

from sklearn import datasets
from utils import LaplacianKernel, get_distance_matrices
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
import numpy as np


# importing the data
dataset = datasets.load_digits()
X = dataset.data
y = dataset.target

# only taking two classes and re-labeling to -1 and 1
y_idx = (y == 0)|(y == 1)
y = y[y_idx]
X = X[y_idx]
y[y == 0] = -1

# splitting the data into train/test/validation (.60, .20, .20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

# calculating distance matrices
d_train, d_test, d_val = get_distance_matrices('digits', mats=[X_train, X_test, X_val])

# validation for laplacian kernel model
# grid search for an optimal value of t
accuracies = []
grid_search = [0.0001, 0.001, 0.01, 0.1, 1, 10]
for i in grid_search:
    model = LaplacianKernel(t=i)
    model.fit(X_train, y_train, distances=d_train)
    prediction = model.predict(X_val, distances=d_val)
    compare = sum(prediction == y_val)
    accuracy = compare / len(y_val)
    accuracies.append(accuracy)

idx = np.argmax(accuracies)
i = grid_search[idx]
model = LaplacianKernel(t=i)
model.fit(X_train, y_train, distances=d_train)
prediction = model.predict(X_val, distances=d_val)
compare = sum(prediction == y_val)
accuracy = compare / len(y_val)
accuracies.append(accuracy)
print(i)
print(accuracy)
print(mean_squared_error(y_val, prediction))
print(classification_report(y_val, prediction))

# creating laplacian kernel model, fitting and predicting
model = LaplacianKernel(t=i)
model.fit(X_train, y_train, distances=d_train)
prediction = model.predict(X_test, distances=d_test)
compare = sum(prediction == y_test)
accuracy = compare / len(y_test)
print(accuracy)
print(mean_squared_error(y_test, prediction))
print(classification_report(y_test, prediction))

# validation for gaussian kernel model
# grid search for an optimal value of t
accuracies = []
grid_search = [0.0001, 0.001, 0.01, 0.1, 1, 10]
for i in grid_search:
    model = LaplacianKernel(n=0.5, t=i)
    model.fit(X_train, y_train, distances=d_train)
    prediction = model.predict(X_val, distances=d_val)
    compare = sum(prediction == y_val)
    accuracy = compare / len(y_val)
    accuracies.append(accuracy)

idx = np.argmax(accuracies)
i = grid_search[idx]
model = LaplacianKernel(n=0.5, t=i)
model.fit(X_train, y_train, distances=d_train)
prediction = model.predict(X_val, distances=d_val)
compare = sum(prediction == y_val)
accuracy = compare / len(y_val)
accuracies.append(accuracy)
print(i)
print(accuracy)
print(mean_squared_error(y_val, prediction))
print(classification_report(y_val, prediction))

# creating gaussian kernel model, fitting and predicting
model = LaplacianKernel(n=0.5, t=i)
model.fit(X_train, y_train, distances=d_train)
prediction = model.predict(X_test, distances=d_test)
compare = sum(prediction == y_test)
accuracy = compare / len(y_test)
print(accuracy)
print(mean_squared_error(y_test, prediction))
print(classification_report(y_test, prediction))
