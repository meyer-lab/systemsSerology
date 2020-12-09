""" Regression methods. """
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import scale
from scipy.stats import pearsonr
from glmnet_python import cvglmnet, cvglmnetCoef
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
    Y = df[function]
    X = df.drop(["subject"] + functions, axis=1)
    Y = Y.to_numpy()
    X = X.to_numpy()
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


def RegressionHelper(X, Y):
    """ Function with common Logistic regression methods. """
    scores = []
    # separate dataframes
    X = scale(X)
    cvfit = cvglmnet(x=X.copy(), y=Y.copy(), nfolds=10, alpha=.8, keep=True, ptype='mse', standardize=False)
    foldid = cvfit["foldid"]
    reps = 100
    for _ in range(reps):
        random.shuffle(foldid)
        cvfit = cvglmnet(x=X.copy(), y=Y.copy(), nfolds=10, alpha=.8, keep=True, foldid=foldid, ptype='mse', standardize=False)
        Y_pred = cvfit['fit_preval'][:, np.where(cvfit["lambdau"] == cvfit['lambda_1se'])[0][0]]
        scores.append([pearsonr(Y, Y_pred)[0], Y_pred, cvfit])

    # TODO: Note that the accuracy on cross-validation is slightly lower than what glmnet returns.
    # score vs. accuracy_score(Y, Y_pred)
    return sorted(scores)[reps//2][1], cvglmnetCoef(sorted(scores)[reps//2][2])