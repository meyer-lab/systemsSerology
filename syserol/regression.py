""" Regression methods. """
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, ElasticNet, LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from .dataImport import (
    importFunction,
    functions,
    importAlterDF,
    AlterIndices,
)


def function_elastic_net(function="ADCC"):
    """ Predict functions using elastic net according to Alter methods"""
    # Import Luminex, Luminex-IGG, Function, and Glycan into DF
    df = importAlterDF(function=True, subjects=False).dropna()

    # separate dataframes
    Y = df[function]
    X = df.drop(["subject"] + functions, axis=1)

    # perform regression
    Y_pred, coef = RegressionHelper(X, Y)
    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred)), coef


def function_prediction(tensorFac, function="ADCC", evaluation="all"):
    """ Predict functions using our decomposition and regression methods"""
    func, _ = importFunction()

    Y = func[function]
    X = tensorFac[1][0]  # subjects x components matrix
    idx = np.zeros(Y.shape, dtype=np.bool)
    idx[AlterIndices()] = 1

    idx = idx[np.isfinite(Y)]
    X = X[np.isfinite(Y), :]
    Y = Y[np.isfinite(Y)]

    # Perform Regression
    Y_pred, coef = RegressionHelper(X, Y)

    if evaluation == "all":
        Y, Y_pred = Y, Y_pred
    elif evaluation == "Alter":
        Y, Y_pred = Y[idx], Y_pred[idx]
    elif evaluation == "notAlter":
        Y, Y_pred = Y[~idx], Y_pred[~idx]
    else:
        raise ValueError("Bad evaluation selection.")

    assert Y.shape == Y_pred.shape
    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred)), coef


def RegressionHelper(X, Y):
    """ Function with common Logistic regression methods. """
    regr = ElasticNetCV(normalize=True, max_iter=10000, cv=20, n_jobs=-1, l1_ratio=0.8).fit(X, Y)
    enet = ElasticNet(alpha=regr.alpha_, l1_ratio=regr.l1_ratio_, normalize=True, max_iter=10000)
    Y_pred = cross_val_predict(enet, X, Y, cv=40, n_jobs=-1)
    return Y_pred, enet.fit(X, Y).coef_
