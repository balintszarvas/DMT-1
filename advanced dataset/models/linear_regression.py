""" This file have all the linear regression models and decision tree regression"""
import numpy as np

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt


import pandas as pd




def polynomial(X, degree = 3):

    ''' Does the polynomial regression

    Inputs:

    degree : Int, degree of the polynomial

    Returns:

    X_train: 2D array, training features

    '''


    poly_features = PolynomialFeatures(degree=degree)

    return poly_features.fit_transform(X_train)

def gaussian(X):
    """
    Computes the data in gaussian activation
    """
    return None
    
def logistic_sigmoid(X):
    """
    Computes the data with logistic sigmoid
    """
    return None


def linear(X):
    """
    Computes the data with linear
    """
    return X



def linear_regression(X_train, y_train,ridge = False, **kwargs):

    """ Does linear regression with a specific degree

        Parameters
    ----------

    X_train : Training features.        

    y_train : Training labels.

    ridge: If use ridge regression,and in that case, input alpha


    Returns
    -------
    model : model for prediction
    """

    if ridge == True:
        model = Ridge(alpha=aplha)

    model = LinearRegression()

    model.fit(X_train, y_train)

    return model




def predict_linear_regression(X_test, model):

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

    """Compute accuracy of prediction using means squared error
    

    Parameters
    ----------

    y_pred : Predicted labels.

    y_test : True labels.
    """

    return np.mean((y_test - y_pred) ** 2)




def train_model(df_train,df_test, target,ridge = False, basis = linear, k_fold = 5, alpha = 1):

    """Train a specified model
    

    Inputs:

    df: Dataframe. Note, Id column will be dropped

    target: Target column
    model: type of function to use in the linear regression

    k-fold : Fraction of the data used for training.


    Returns:

    accuracy: accuracy fo the model 
    model: training model
    """
    X, y = df.drop(target, axis = 1), df_train[target]
    accuracy = []
    
    if k_fold == False:
        kf = LeaveOneOut()
    else:
        kf = KFold(n_splits= k_fold, shuffle = True)

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        
        X_train, X_test = basis(X.iloc[train_index]), basis(X.iloc[test_index])
        y_train, y_test = y[train_index], y[test_index]   

        model = linear_regression(X_train, y_train, ridge = ridge, alpha = alpha)

        y_pred = predict_linear_regression(X_test, model)
        accuracy.append(compute_accuracy(y_pred, y_test))
    

    return model, accuracy, np.mean(accuracy)

def plotter(parameters, mean_accuracies):
    plt.plot(parameters,mean_accuracies)
    plt.xlabel("")
    plt.ylabel("k_fold")
    plt.grid(True)
    plt.show()

def plot_predictions(y_pred, y):

    fig, axs = plt.subplots(1,1)
    axs.plot(y_pred, label = "Predicted", color = 'blue')
    axs.plot(y, label = "Original", color = 'green')
    plt.title(compute_accuracy(y_pred, y))
    axs.grid(True)
    plt.show()



if __name__ == '__main__':
    
    df = pd.read_csv('../DMT-1/advanced dataset/data/train.csv')
    columns_to_keep = ['mood','activity', 'appCat.other', 'appCat.social', 'circumplex.valence']


    columns_to_drop = set(df.columns) - set(columns_to_keep)

    df = df.drop(columns=columns_to_drop)
    k_fold = 5

    model, accuracy, mean_accuracy,y_pred,y = train_model(df_train, df_test, target='mood', k_fold = False)
    plot_predictions(y_pred,y)











