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
    df = importAlterDF(function=True, subjects=False)
    df_merged = df.dropna()
    # separate dataframes
    Y = df_merged[function]
    df_variables = df_merged.drop(["subject"] + functions, axis=1)

    # perform regression
    regr = ElasticNetCV(normalize=True, max_iter=10000, cv=30, n_jobs=-1, l1_ratio=0.8)
    regr.fit(df_variables, Y)
    enet = ElasticNet(
        alpha=regr.alpha_, l1_ratio=regr.l1_ratio_, normalize=True, max_iter=10000
    )

    Y_pred = cross_val_predict(enet, df_variables, Y, cv=len(Y), n_jobs=-1)
    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred)), regr.coef_

def Alter_weights():
    """ Analyze Coefficients used in Alter Function Prediction"""
    df = importAlterDF(function=True, subjects=False)
    df_merged = df.dropna()
    df_variables = df_merged.drop(["subject"] + functions, axis=1)

    #Gather Coefficients
    df_new = pd.DataFrame(columns=df_variables.columns)
    for i in functions:
        _, _, _, coef = function_elastic_net(i)
        df_new = df_new.append(pd.DataFrame(coef.reshape(1, -1), columns=list(df_variables)))
    df_new.index = functions

    df_coefs = df_new.loc[:, (df_new != 0).any(axis=0)] # All functions' weighted coefficients
    return df_coefs

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
    regr = ElasticNetCV(normalize=True, max_iter=10000, cv=30, n_jobs=-1, l1_ratio=0.8)
    regr.fit(X, Y)
    enet = ElasticNet(alpha=regr.alpha_, l1_ratio=regr.l1_ratio_, normalize=True, max_iter=10000)
    Y_pred = cross_val_predict(enet, X, Y, cv=len(Y), n_jobs=-1)

    if evaluation == "all":
        Y, Y_pred = Y, Y_pred
    elif evaluation == "Alter":
        Y, Y_pred = Y[idx], Y_pred[idx]
    elif evaluation == "notAlter":
        Y, Y_pred = Y[~idx], Y_pred[~idx]
    else:
        raise ValueError("Bad evaluation selection.")

    assert Y.shape == Y_pred.shape
    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))
