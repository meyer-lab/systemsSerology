""" Regression methods. """
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from .dataImport import (
    importFunction,
    functions,
    importAlterDF,
    AlterIndices,
)
from glmnet import ElasticNet


def function_elastic_net(function="ADCC"):
    """ Predict functions using elastic net according to Alter methods"""
    # Import Luminex, Luminex-IGG, Function, and Glycan into DF
    df = importAlterDF(function=True, subjects=False).dropna()

    # separate dataframes
    Y = df[function]
    X = df.drop(["subject"] + functions, axis=1)

    # perform regression
    scores = []
    for _ in range(100):
        Y_pred, coef = RegressionHelper(X, Y)
        scores.append([np.sqrt(r2_score(Y, Y_pred)), coef])
    scores.sort(key=lambda x: x[0])
    return Y, Y_pred, scores[49][0], scores[49][1]


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
    scores = []
    for _ in range(100):
        Y_pred, coef = RegressionHelper(X, Y)

        if evaluation == "Alter":
            Y, Y_pred = Y[idx], Y_pred[idx]
        elif evaluation == "notAlter":
            Y, Y_pred = Y[~idx], Y_pred[~idx]
        elif evaluation != "all":
            raise ValueError("Bad evaluation selection.")
        assert Y.shape == Y_pred.shape
        scores.append([np.sqrt(r2_score(Y, Y_pred)), coef])
    scores.sort(key=lambda x: x[0])
    return Y, Y_pred, scores[49][0], scores[49][1]


def RegressionHelper(X, Y):
    """ Function with common Logistic regression methods. """
    enet = ENet(alpha=.8, n_splits=10, n_jobs=25, scoring="mean_squared_error").fit(X, Y)
    Y_pred = enet.predict(X)
    return Y_pred, enet.coef_
