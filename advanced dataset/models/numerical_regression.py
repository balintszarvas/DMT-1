""" This file have all the linear regression models and decision tree regression"""
import numpy as np

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

import pandas as pd



def load_dataset(df, target, split = 0.05):

    """Load and split the dataset into training and test parts.
    

    Inputs:

    df: Dataframe. Note, Id column will be dropped

    Target: Target column

    k-fold : Fraction of the data used for training.
    

    Returns:

    X_train : Training features.

    y_train : Training labels.

    X_test : Test features.

    y_test : Test labels.
    """


    X, y = df.drop(target), df[target]

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=123, test_size=(1 - split))

    return X_train, X_test, y_train, y_test



def polynomial(degree = 3):

    ''' Does the polynomial regression

    Inputs:

    degree : Int, degree of the polynomial

    Returns:

    X_train: 2D array, training features

    '''


    poly_features = PolynomialFeatures(degree=degree)

    return poly_features.fit_transform(X_train)


def linear_regression(X_train, y_train, model = 'linear', **kwargs):

    """ Does linear regression with a specific degree

        Parameters
    ----------

    X_train : Training features.        

    y_train : Training labels.

    X_test : Test features.

    Returns
    -------
    model : model for prediction
    """

    if model == 'polynomial':

        X_train = polynomial()


    model = LinearRegression()

    model.fit(X_train, y_train)

    return model




def predict_linear_regression(X_train, X_test, model):

    """Generate predictions for all points in the test set.
    

    Parameters
    ----------

    X_train : Training features.        

    y_train : Training labels.

    X_test : Test features.

    Returns
    -------

    y_pred : array, shape (N_test)

        Predictions for the test data.
    """


    y_pred = model.predict(X_test)
    return y_pred



def compute_accuracy(y_pred, y_test):

    """Compute accuracy of prediction.
    

    Parameters
    ----------

    y_pred : Predicted labels.

    y_test : True labels.
    """

    return np.mean(y_pred == y_test)




def train_model(df, target, type = 'linear regression', k_fold = 23):

    """Train a specified model
    

    Inputs:

    df: Dataframe. Note, Id column will be dropped

    target: Target column
    type: type of mode l to train

    k-fold : Fraction of the data used for training.


    Returns:

    accuracy: accuracy fo the model 
    model: training model
    """
    X, y = df.drop(target), df[target]
    accuracy = []
    
    if k_fold == False:
        kf = LeaveOneOut()
    else:
        kf = KFold(n_splits= k_fold)

    for i, (train_index, test_index) in enumerate(kf.split(X)):

        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index],y[test_index]

        model = linear_regression(X_train, y_train, model = 'linear')

        y_pred = predict_linear_regression(X_train, X_test, model)
        accuracy.append(compute_accuracy(y_pred, y_test))
    
    model = linear_regression(X, y, model = 'linear')

    return model, accuracy





