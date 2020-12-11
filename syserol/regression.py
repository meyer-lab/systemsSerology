""" Regression methods. """
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from scipy.stats import pearsonr
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

    # perform regression
    Y_pred, coef = RegressionHelper(X, Y)
    return Y, Y_pred, pearsonr(Y, Y_pred)[0], coef


def function_prediction(tensorFac, function="ADCC", evaluation="all"):
    """ Predict functions using our decomposition and regression methods"""
    func, _ = importFunction()

    Y = func[function]
    subset = np.isfinite(Y)
    X = tensorFac[1][0][subset, :]  # subjects x components matrix
    Y = Y[subset]

    # Perform Regression
    Y_pred, coef = RegressionHelper(X, Y)
    Y, Y_pred = selectAlter(Y, Y_pred, evaluation, subset=subset)

    return Y, Y_pred, pearsonr(Y, Y_pred)[0], coef


def RegressionHelper(X, Y, classify=False):
    """ Function with the regression cross-validation strategy. """
    if X.shape[1] < 20:
        return RegressionHelperGP(X, Y, classify=classify)

    if classify:
        X = scale(X)
        est = LogisticRegressionCV(penalty="elasticnet", solver="saga")
    else:
        est = ElasticNetCV(normalize=True)

    est.l1_ratios = [0.8]
    est.cv = 10
    est.max_iter = 10000

    est = est.fit(X, Y)
    coef = np.squeeze(est.coef_)

    Y_pred = cross_val_predict(est, X, Y, cv=20, n_jobs=-1)
    return Y_pred, coef


def RegressionHelperGP(X, Y, classify=False):
    """ Function with the regression cross-validation strategy. """
    kern = ConstantKernel() * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(noise_level_bounds=(0.1, 10))

    if classify:
        X = scale(X)
        estG = GaussianProcessClassifier(kern, warm_start=True, n_restarts_optimizer=3)
    else:
        estG = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=3)

    Y_pred = cross_val_predict(estG, X, Y, cv=20, n_jobs=-1)
    estG.fit(X, Y)
    coef = np.zeros(X.shape[1])

    return Y_pred, coef
