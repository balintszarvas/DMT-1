""" This file have all the linear regression models and decision tree regression"""
import numpy as np

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt
import scipy.stats as stats



import pandas as pd


### BASIS FUNCTIONS

def polynomial(X, degree = 3):

    ''' Does the polynomial regression

    Inputs:

    degree : Int, degree of the polynomial

    Returns:

    X_train: 2D array, training features

    '''


    poly_features = PolynomialFeatures(degree=degree)

    return poly_features.fit_transform(X)

def linear(X):
    """
    Computes the data with linear
    """
    return X


def gaussian(X):
    """
    Computes teh data with gaussian transformations. NOT YET IMPLEMENTED, ONLY IF LINEAR OR PLYNOMAIL DOES NOT WORK.
    """
    return None
    
def logistic_sigmoid(X):
    """
    Computes the data with logistic sigmoid. NOT YET IMPLEMENTED, ONLY IF LINEAR OR PLYNOMAIL DOES NOT WORK.
    """
    return None

 ### SUPPORT FUNCTIONS
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



### MODEL TRAINING

def train_model(df, target,ridge = False, basis = linear, k_fold = 5, alpha = 1):

    """Train a linear regression model. This is the relevant functions, the other ones are support function
    

    Parameters:
    -----------

    df: Dataframe. Note, Id column will be dropped

    target: Target column

    model: type of function to use in the linear regression. Can be linear, gaussian, polynomial or logistic

    k-fold : Fraction of the data used for training, if false uses LOOCV

    alpha: For ridge regression, lamba in Overleaf


    Returns:
    -----------

    accuracy: accuracy of the model all trained models 
    model: training model
    mean_accuracy: The mean accuracy of all models
    y_pred,y_test: last prediction and test done

    """
    X, y = df.drop(target, axis = 1), df[target]
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
    

    return model, accuracy, np.mean(accuracy), y_pred, y_test

def plot_residuals(y_pred, y_test, compare = True):
    """Does the residuals plots:
    Parameters:
    -----------
    y_pred: predicted data
    y_test:actual data
    """
    
    residuals = y - y_pred


    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=10, density=True, alpha=0.6, color='blue', edgecolor='black')
    if compare == True:
        mu, sigma = np.mean(residuals), np.std(residuals)
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, mu, sigma)
        plt.plot(x, p, 'k', linewidth=2)
        plt.title('Residuals Histogram and Normal Distribution')
        plt.xlabel('Residuals (Actual - Predicted)')
        plt.ylabel('Density')
        plt.legend(['Normal Distribution', 'Residuals'])

    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    
    df = pd.read_csv('../DMT-1/train.csv')
    columns_to_keep = ['mood','activity', 'appCat.other', 'appCat.social', 'circumplex.valence']


    columns_to_drop = set(df.columns) - set(columns_to_keep)

    df = df.drop(columns=columns_to_drop)
    k_fold = 5

    model, accuracy, mean_accuracy,y_pred,y_test = train_model(df, target='mood', k_fold = 10)
    print(y_pred)
    print(y_test)
    plot_residuals(y_pred,y_test)











