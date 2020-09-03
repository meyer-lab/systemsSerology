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


def elasticNetFunc(X, Y):
    """ Function with common elastic net methods. """
    regr = ElasticNetCV(normalize=True, max_iter=10000, cv=30, n_jobs=-1)
    regr.fit(X, Y)
    enet = ElasticNet(alpha=regr.alpha_, l1_ratio=regr.l1_ratio_, normalize=True, max_iter=10000)
    Y_pred = cross_val_predict(enet, X, Y, cv=30, n_jobs=-1)
    rsq = np.sqrt(r2_score(Y, Y_pred))
    return Y_pred, rsq


def function_elastic_net(function="ADCC"):
    """ Predict functions using elastic net according to Alter methods"""
    # Import Luminex, Luminex-IGG, Function, and Glycan into DF
    df = importAlterDF(function=True, subjects=False)
    df_merged = df.dropna()
    # separate dataframes
    df_func = df_merged[functions]
    df_variables = df_merged.drop(["subject"] + functions, axis=1)

    # perform regression
    Y = df_func[function]
    Y_pred, _ = elasticNetFunc(df_variables, Y)

    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))


def function_prediction(tensorFac, function="ADCC", evaluation="all", enet=True):
    """ Predict functions using our decomposition and regression methods"""
    func, _ = importFunction()

    Y = func[function]
    X = tensorFac[1][0]  # subjects x components matrix
    dropped = np.nonzero(np.isnan(Y.to_numpy()))
    X = X[np.isfinite(Y), :]
    Y = Y[np.isfinite(Y)]

    if enet is True:
        Y_pred, _ = elasticNetFunc(X, Y)
    else:
        Y_pred = cross_val_predict(SVR(), X, Y, cv=10, n_jobs=-1)

    if evaluation == "all":
        return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))
    elif evaluation == "Alter":
        return accuracy_alterSubj(Y, Y_pred, dropped[0])
    elif evaluation == "notAlter":
        return accuracy_alterSubj(Y, Y_pred, dropped[0], union=False)
    else:
        raise ValueError("Wrong selection for evaluation.")


def accuracy_alterSubj(Y, Ypred, dropped, union=True):
    """ Calculate the Accuracy for Only Subjects Included in Alter """
    indices = AlterIndices()

    # Inflate back to original size
    Ypred = np.insert(Ypred, dropped, np.nan)
    Y = np.insert(Y.to_numpy(), dropped, np.nan)

    if union is True:
        # Reduce to Alter subjects
        Ypred = Ypred[indices]
        Y = Y[indices]
    else:
        # Remove Alter cases
        Ypred = np.delete(Ypred, indices)
        Y = np.delete(Y, indices)

    # Remove any missing cases
    Ypred = Ypred[np.isfinite(Y)]
    Y = Y[np.isfinite(Y)]

    return Y, Ypred, np.sqrt(r2_score(Y, Ypred))
