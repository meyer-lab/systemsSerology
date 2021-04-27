""" Regression methods. """
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.utils import resample as resampleSK
from sklearn.preprocessing import scale
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV, LogisticRegression, ElasticNet
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.stats import pearsonr
from .dataImport import (
    importFunction,
    functions,
    importAlterDF,
    selectAlter,
)


def function_elastic_net(function="ADCC", resample=False):
    """ Predict functions using elastic net according to Alter methods"""
    # Import Luminex, Luminex-IGG, Function, and Glycan into DF
    df = importAlterDF(function=True, subjects=False).dropna()

    # separate dataframes
    Y = df[function]
    X = df.drop(["subject"] + functions, axis=1)

    # perform regression
    Y_pred, coef, X, Y = RegressionHelper(X, Y, resample=resample)
    return Y, Y_pred, pearsonr(Y, Y_pred)[0], coef


def function_prediction(Xin, function="ADCC", **kwargs):
    """ Predict functions using our decomposition and regression methods"""
    func, _ = importFunction()

    Y = func[function]
    subset = np.isfinite(Y)
    X = Xin[subset, :]  # subjects x components matrix
    Y = Y[subset]
    assert Y.dtype == float

    # Perform Regression
    Y_pred, coef, X, Y = RegressionHelper(X, Y, **kwargs)
    accuracy = pearsonr(*selectAlter(Y, Y_pred, subset))[0]

    return Y, Y_pred, accuracy, coef


def make_regression_df(X, resample=False):
    """ Make the dataframe of prediction accuracies. """
    # Gather Function Prediction Accuracies
    accuracies = []
    accuracies += [function_prediction(X, resample=resample, function=f)[2] for f in functions]
    accuracies += [function_elastic_net(f, resample=resample)[2] for f in functions]
    accuracies += [function_prediction(X, function=f, randomize=True)[2] for f in functions]

    # Create DataFrame
    model = ["CMTF"] * 6 + ["Alter et al"] * 6 + ["Randomized"] * 6
    function = functions + functions + functions
    data = {"Accuracy": accuracies, "Model": model, "Function": function}
    return pd.DataFrame(data)


def RegressionHelper(X, Y, randomize=False, resample=False):
    """ Function with the regression cross-validation strategy. """
    X = np.copy(X)
    Y = np.copy(Y)

    if randomize:
        np.random.shuffle(X)

    if resample:
        X, Y = resampleSK(X, Y)

    if Y.dtype == int:
        X = scale(X)
        estCV = LogisticRegressionCV(penalty="elasticnet", solver="saga", cv=10, l1_ratios=[0.8], n_jobs=-1, max_iter=1000000)
        estCV.fit(X, Y)
        est = LogisticRegression(C=estCV.C_[0], penalty="elasticnet", solver="saga", l1_ratio=0.8, max_iter=1000000)
        cv = StratifiedKFold(n_splits=10, shuffle=True)
    else:
        assert Y.dtype == float
        estCV = ElasticNetCV(normalize=True, l1_ratio=0.8, cv=10, n_jobs=-1, max_iter=1000000)
        estCV.fit(X, Y)
        est = ElasticNet(normalize=True, alpha=estCV.alpha_, l1_ratio=0.8, max_iter=1000000)
        cv = KFold(n_splits=10, shuffle=True)

    est = est.fit(X, Y)
    coef = np.squeeze(est.coef_)
    Y_pred = cross_val_predict(est, X, Y, cv=cv, n_jobs=-1)

    return Y_pred, coef, X, Y
