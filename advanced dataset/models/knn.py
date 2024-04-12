

import numpy as np
from sklearn import datasets, model_selection
import matplotlib.pyplot as plt

"""
This file contains all K-NN functions
"""

def load_dataset(df, target, var, split):
    """Load and split the dataset into training and test parts.
    
    Inputs:
    -split : Fraction of the data used for training.
    
    Returns:
    X_train : Training features.
    y_train : Training labels.
    X_test : Test features.
    y_test : Test labels.
    """

    X, y = df['data'], df['target']
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=123, test_size=(1 - split))
    return X_train, X_test, y_train, y_test

def euclidean_distance(x1, x2):
    """Compute pairwise Euclidean distances between two data points.
    
    Parameters
    ----------
    x1 : First set of data points.
    x2 : Second set of data points.
    
    Returns
    -------
    distance : float array, shape (N, M)
        Pairwise Euclidean distances between x1 and x2.
    """
    return np.sqrt(np.sum((x1[:, None] - x2[None])**2, -1))


def get_neighbors_labels(X_train, y_train, X_new, k):
    """Get the labels of the k nearest neighbors of the datapoints x_new.
    
    Parameters
    ----------
    X_train : Training features.
    y_train : Training labels.
    X_new : Data points for which the neighbors have to be found.
    k : Number of neighbors to return.
        
    Returns
    -------
    neighbors_labels : array, shape (M, k)
        Array containing the labels of the k nearest neighbors.
    """
    distances = euclidean_distance(X_new, X_train)
    nearest = np.argsort(distances, axis=1)[:, :k]
    
    return y_train[nearest]

def get_response(neighbors_labels, num_classes=3):
    """Predict label given the set of neighbors.
    
    Parameters
    ----------
    neighbors_labels : Array containing the labels of the k nearest neighbors per data point.
    num_classes : Number of classes in the dataset.
    
    Returns
    -------
    y :
        Majority class among the neighbors.
    """
    class_votes = (neighbors_labels[:, :, None] == np.arange(num_classes)[None, None]).sum(1)
    return np.argmax(class_votes, 1)


def compute_accuracy(y_pred, y_test):
    """Compute accuracy of prediction.
    
    Parameters
    ----------
    y_pred : Predicted labels.
    y_test : True labels.
    """
    return np.mean(y_pred == y_test)


    
def predict(X_train, y_train, X_test, k):
    """Generate predictions for all points in the test set.
    
    Parameters
    ----------
    X_train : Training features.        
    y_train : Training labels.
    X_test : Test features.
    k : Number of neighbors to consider.
    
    Returns
    -------
    y_pred : array, shape (N_test)
        Predictions for the test data.
    """
    neighbors = get_neighbors_labels(X_train, y_train, X_test, k)
    y_pred = get_response(neighbors)
    return y_pred

