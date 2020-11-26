""" Regression methods. """
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_predict
from glmnet import ElasticNet
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
    X = tensorFac[1][0][np.isfinite(Y), :]  # subjects x components matrix
    idx = np.zeros(Y.shape, dtype=np.bool)
    idx[AlterIndices()] = 1

    idx = idx[np.isfinite(Y)]
    Y = Y[np.isfinite(Y)]

    # Perform Regression
    Y_pred, coef = RegressionHelper(X, Y)

    if evaluation == "Alter":
        Y, Y_pred = Y[idx], Y_pred[idx]
    elif evaluation == "notAlter":
        Y, Y_pred = Y[~idx], Y_pred[~idx]
    elif evaluation != "all":
        raise ValueError("Bad evaluation selection.")

    assert Y.shape == Y_pred.shape
    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred)), coef


def RegressionHelper(X, Y):
    """ Function with common Logistic regression methods. """
    glmnet = ElasticNet(alpha = .8, n_jobs=10, n_splits=10).fit(X, Y)
    coef = glmnet.coef_

    Y_pred = cross_val_predict(glmnet, X, Y, cv=10, n_jobs=-1)

    # TODO: Note that the accuracy on cross-validation is slightly lower than what glmnet returns.
    # score vs. accuracy_score(Y, Y_pred)
    return Y_pred, coef