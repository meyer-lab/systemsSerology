""" Regression methods. """
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from .dataImport import (
    importFunction,
    functions,
    importAlterDF,
    getAxes,
    AlterIndices,
)


def function_elastic_net(function="ADCC"):
    """ Predict functions using elastic net according to Alter methods"""
    # Import Luminex, Luminex-IGG, Function, and Glycan into DF
    df = importAlterDF()
    func, _ = importFunction()
    df_merged = df.merge(func, on="subject", how="inner")
    df_merged = df_merged.dropna()
    # separate dataframes
    df_func = df_merged[functions]
    df_variables = df_merged.drop(["subject"] + functions, axis=1)

    # perform regression
    Y = df_func[function]
    Y_pred = elastic_net_helper(df_variables, Y)

    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))


def function_prediction(tensorFac, function="ADCC", evaluation="all", enet=True):
    """ Predict functions using our decomposition and regression methods"""
    func, _ = importFunction()

    Y = func[function]
    X = tensorFac[1][0]  # subjects x components matrix
    dropped = np.argwhere(np.isnan(Y))
    X = X[np.isfinite(Y), :]
    Y = Y[np.isfinite(Y)]

    if enet is True:
        Y_pred = elastic_net_helper(X, Y)
    else:
        Y_pred = cross_val_predict(SVR(), X, Y, cv=10)

    if evaluation == "all":
        return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))
    elif evaluation == "Alter":
        return accuracy_alterSubjOnly(Y, Y_pred, dropped)
    elif evaluation == "notAlter":
        return accuracy_leftoutAlterSubj(Y, Y_pred, dropped)
    else:
        raise ValueError("Wrong selection for evaluation.")



def elastic_net_helper(X, Y):
    """ Helper Function for Elastic Net Regression"""
    regr = ElasticNetCV(normalize=True, max_iter=10000, cv=10)
    regr.fit(X, Y)
    enet = ElasticNet(alpha=regr.alpha_, normalize=True, max_iter=10000)
    Y_pred = cross_val_predict(enet, X, Y, cv=10)
    return Y_pred


def accuracy_alterSubjOnly(Y, Ypred, dropped):
    """ Calculate the Accuracy for Only Subjects Included in Alter """
    indices = AlterIndices()

    # Inflate back to original size
    Ypred = np.insert(Ypred, dropped, np.nan)
    Y = np.insert(Y, dropped, np.nan)

    # Reduce to Alter subjects
    Ypred = Ypred[indices]
    Y = Y[indices]

    # Remove any missing cases
    Ypred = Ypred[np.isfinite(Y)]
    Y = Y[np.isfinite(Y)]

    return Y, Ypred, np.sqrt(r2_score(Y, Ypred))


def accuracy_leftoutAlterSubj(Y, Ypred, dropped):
    """ Calculate accuracy for subjects that Alter could not predict and we did predict"""
    indices = AlterIndices()
    
    # Inflate back to original size
    Ypred = np.insert(Ypred, dropped, np.nan)
    Y = np.insert(Y, dropped, np.nan)

    # Remove Alter cases
    Ypred = Ypred.delete(indices)
    Y = Y.delete(indices)

    # Remove any missing cases
    Ypred = Ypred[np.isfinite(Y)]
    Y = Y[np.isfinite(Y)]

    return Y, Ypred, np.sqrt(r2_score(Y, Ypred))
