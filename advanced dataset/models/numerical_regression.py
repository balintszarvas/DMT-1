""" This file have all the liner regression models
IN PROGRESS"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd


def load_dataset(df, target, split = 0.2):
    """Load and split the dataset into training and test parts.
    
    Inputs:
    -df: Dataframe. Note, Id column will be dropped
    -Target: Target column
    -split : Fraction of the data used for training.
    
    Returns:
    X_train : Training features.
    y_train : Training labels.
    X_test : Test features.
    y_test : Test labels.
    """

    X, y = df.drop(target), df[target]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=123, test_size=(1 - split))
    return X_train, X_test, y_train, y_test


def linear_regression(X_train, y_train, degree = 4):
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
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X_train)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    return model

def predict_linear_regression(X_train_poly, X_test, model):
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

    y_pred = model.predict(X_test_poly)
    return y_pred


def optimize(df, target, split, m = 10):
    """ Optimizes the linear regression model and plots the results
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
    return None

