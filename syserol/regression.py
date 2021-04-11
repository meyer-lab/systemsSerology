""" Regression methods. """
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.utils import resample as resampleSK
from sklearn.preprocessing import scale
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV, LogisticRegression, ElasticNet
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
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
    preds = [function_prediction(X, resample=resample, function=f)[2] for f in functions]

    accuracies = [function_elastic_net(f, resample=resample)[2] for f in functions]
    baselines = [function_prediction(X, function=f, randomize=True)[2] for f in functions]
    accuracies = accuracies + preds + baselines

    # Create DataFrame
    model = ["TMTF"] * 6 + ["Alter et al"] * 6 + ["Randomized"] * 6
    function = functions + functions + functions
    data = {"Accuracy": accuracies, "Model": model, "Function": function}
    functions_df = pd.DataFrame(data)  # Function Prediction DataFrame, Figure 5A
    return functions_df


def RegressionHelper(X, Y, randomize=False, resample=False):
    """ Function with the regression cross-validation strategy. """
    X = np.copy(X)
    Y = np.copy(Y)
    kern = ConstantKernel() * RBF(np.ones(X.shape[1]), (1e-2, 1e14))
    kern += WhiteKernel(noise_level_bounds=(0.001, 0.8))

    if randomize:
        np.random.shuffle(X)

    if resample:
        X, Y = resampleSK(X, Y)

    if Y.dtype == int:
        X = scale(X)
        estCV = LogisticRegressionCV(penalty="elasticnet", solver="saga", cv=10, l1_ratios=[0.8], n_jobs=-1, max_iter=1000000)
        estCV.fit(X, Y)
        est = LogisticRegression(C=estCV.C_[0], penalty="elasticnet", solver="saga", l1_ratio=0.8, max_iter=1000000)
        estG = GaussianProcessClassifier(kern, n_restarts_optimizer=5)
        cv = StratifiedKFold(n_splits=10, shuffle=True)
    else:
        assert Y.dtype == float
        estCV = ElasticNetCV(normalize=True, l1_ratio=0.8, cv=10, n_jobs=-1, max_iter=1000000)
        estCV.fit(X, Y)
        est = ElasticNet(normalize=True, alpha=estCV.alpha_, l1_ratio=0.8, max_iter=1000000)
        estG = GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=5)
        cv = KFold(n_splits=10, shuffle=True)

    est = est.fit(X, Y)
    coef = np.squeeze(est.coef_)
    Y_pred = cross_val_predict(est, X, Y, cv=cv, n_jobs=-1)

    # TODO: Get Gaussian process model working for multinomial case
    if (X.shape[1] < 20) and (np.unique(Y).size != 4):
        estG.fit(X, Y)
        coef = np.sign(coef) / estG.kernel_.get_params()["k1__k2__length_scale"]
        estG.optimizer = None
        estG.kernel = estG.kernel_
        Y_pred_G = cross_val_predict(estG, X, Y, cv=cv, n_jobs=-1)

        if Y.dtype == int:
            better = accuracy_score(Y, Y_pred_G) > accuracy_score(Y, Y_pred)
        else:
            better = pearsonr(Y, Y_pred_G)[0] > pearsonr(Y, Y_pred)[0]

        if better:
            return Y_pred_G, coef, X, Y

    return Y_pred, coef, X, Y
