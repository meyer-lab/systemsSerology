""" Regression methods. """
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from glmnet_python import cvglmnet, cvglmnetCoef, cvglmnetPredict
from .dataImport import (
    importFunction,
    functions,
    importAlterDF,
    selectAlter,
)



def function_elastic_net(function="ADCC"):
    """ Predict functions using elastic net according to Alter methods"""
    # Import Luminex, Luminex-IGG, Function, and Glycan into DF
    df = importAlterDF(function=True, subjects=False).dropna()

    # separate dataframes
    Y = df[function].to_numpy()
    X = df.drop(["subject"] + functions, axis=1).to_numpy()

    # perform regression
    Y_pred, coef = RegressionHelper(X, Y)
    return Y, Y_pred, pearsonr(Y, Y_pred)[0], coef


def function_prediction(tensorFac, function="ADCC", evaluation="all"):
    """ Predict functions using our decomposition and regression methods"""
    func, _ = importFunction()

    Y = func[function]
    subset = np.isfinite(Y)
    X = tensorFac[1][0][subset, :]  # subjects x components matrix
    Y = Y[subset].to_numpy()

    # Perform Regression
    Y_pred, coef = RegressionHelper(X, Y)
    Y, Y_pred = selectAlter(Y, Y_pred, evaluation, subset=subset)

    return Y, Y_pred, pearsonr(Y, Y_pred)[0], coef


def RegressionHelper(X, Y, classify=False):
    """ Function with the regression cross-validation strategy. """
    assert Y.ndim == 1
    assert X.shape[0] == Y.size
    assert X.ndim == 2

    if classify:
        kwargs = {"family": "binomial", "ptype": "class"}
    else:
        kwargs = {"ptype": "mse"}

    cvfit = cvglmnet(x=X.copy(), y=Y.copy(), nfolds=20, alpha=.8, standardize=True, **kwargs)
    coef = np.squeeze(cvglmnetCoef(cvfit))[:-1] # remove the intercept
    assert coef.ndim == 1
    assert coef.size == X.shape[1]

    Y_pred = np.empty_like(Y)
    kf = KFold(n_splits=20, shuffle=True)

    for train_i, test_i in kf.split(X):
        cvfit = cvglmnet(x=X[train_i, :].copy(), y=Y[train_i].copy(), nfolds=20, alpha=.8, standardize=True, **kwargs)
        Y_pred[test_i] = np.squeeze(cvglmnetPredict(cvfit, newx = X[test_i, :].copy()))

    if classify:
        Y_pred = Y_pred > 0.5

    return Y_pred, coef
