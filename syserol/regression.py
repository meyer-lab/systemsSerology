""" Regression methods. """
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample as resampleSK
from sklearn.preprocessing import scale
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV, LogisticRegression, ElasticNet
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.feature_selection import SequentialFeatureSelector
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
    functions_df = pd.DataFrame(data)  # Function Prediction DataFrame, Figure 5A
    return functions_df


def RegressionHelper(X, Y, randomize=False, resample=False):
    """ Function with the regression cross-validation strategy. """
    kern = RBF() + WhiteKernel()

    factoredData = X.shape[1] < 15

    if randomize:
        assert factoredData  # Make sure that we're randomizing our model
        X = np.copy(X)
        np.random.shuffle(X)

    if resample:
        X, Y = resampleSK(X, Y)

    if Y.dtype == int:
        X = scale(X)
        cv = StratifiedKFold(n_splits=10, shuffle=True)

        if factoredData:
            est = GaussianProcessClassifier(kern)
        else:
            estCV = LogisticRegressionCV(penalty="elasticnet", solver="saga", cv=cv, l1_ratios=[0.8], n_jobs=-1, max_iter=10000)
            estCV.fit(X, Y)
            est = LogisticRegression(C=estCV.C_[0], penalty="elasticnet", solver="saga", l1_ratio=0.8, max_iter=10000)
    else:
        assert Y.dtype == float
        cv = KFold(n_splits=10, shuffle=True)

        if factoredData:
            est = GaussianProcessRegressor(kern, normalize_y=True)
        else:
            estCV = ElasticNetCV(normalize=True, l1_ratio=0.8, cv=cv, n_jobs=-1, max_iter=10000)
            estCV.fit(X, Y)
            est = ElasticNet(normalize=True, alpha=estCV.alpha_, l1_ratio=0.8, max_iter=10000)

    if factoredData:
        SFS = SequentialFeatureSelector(est, n_features_to_select=2, cv=cv)
        est = make_pipeline(SFS, est)
        est.fit(X, Y)
        coef = est.steps[0][1].support_

        # Set hyperparameters
        est.steps[0][1].estimator.optimizer = None
        est.steps[0][1].estimator.kernel = est.steps[1][1].kernel_
        est.steps[1][1].optimizer = None
        est.steps[1][1].kernel = est.steps[1][1].kernel_
        print("Done with another.")
    else:
        est = est.fit(X, Y)
        coef = np.squeeze(est.coef_)

    Y_pred = cross_val_predict(est, X, Y, cv=cv, n_jobs=-1)

    return Y_pred, coef, X, Y
