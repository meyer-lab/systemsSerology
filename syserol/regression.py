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
    functions,
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
    df_merged = importAlterDF()

    # separate dataframes
    df_func = df_merged[functions]
    df_variables = df_merged.drop(["subject"] + functions, axis=1)

    # perform regression
    Y = df_func[function]
    Y_pred, rsq = elasticNetFunc(df_variables, Y)

    return Y, Y_pred, rsq


def noCMTF_function_prediction(tensorFac, function="ADCC"):
    """ Predict functions using our decomposition and regression methods"""
    func, _ = importFunction()
    df = pd.DataFrame(tensorFac[1][0])  # subjects x components matrix
    df = df.join(func, how="inner")
    df = df.dropna()
    df_func = df[functions]
    df_variables = df.drop(["subject"] + functions, axis=1)

    Y = df_func[function]
    Y_pred, rsq = elasticNetFunc(df_variables, Y)

    return Y, Y_pred, rsq

def SVR_noCMTF_function_prediction(tensorFac, function="ADCC"):
    """ Predict functions using our decomposition and SVR regression methods"""
    func, _ = importFunction()
    df = pd.DataFrame(tensorFac[1][0])  # subjects x components matrix
    df = df.join(func, how="inner")
    df = df.dropna()
    df_func = df[functions]
    df_variables = df.drop(
        ["subject"] + functions, axis=1
    )

    X = df_variables
    Y = df_func[function]
    Y_pred = cross_val_predict(SVR(), X, Y, cv=10, n_jobs=-1)
    accuracy = np.sqrt(r2_score(Y, Y_pred))
    print(f"Accuracy: {accuracy}")

    return Y, Y_pred, accuracy

def ourSubjects_function_prediction(tensorFac, function="ADCC"):
    """ Predict functions for subjects specifically left out of Alter using regression methods"""
    # Re-Create Alter DataFrame with leftout subjects
    df_merged = importAlterDF()

    fullsubj = np.array(df_merged["subject"])  # Subjects only included in Alter
    subjects, _, _ = getAxes()

    # Subjects left out of Alter
    indices = [idx for idx, i in enumerate(subjects) if i not in fullsubj]

    func, _ = importFunction()
    df = pd.DataFrame(tensorFac[1][0])  # subjects x components matrix
    df = df.join(func, how="inner")
    df = df.iloc[indices]
    df = df.dropna()
    df_func = df[functions]
    df_variables = df.drop(["subject"] + functions, axis=1)

    Y = df_func[function]
    Y_pred, _ = elasticNetFunc(df_variables, Y)

    return Y, Y_pred

def SVR_ourSubjects_function_prediction(tensorFac, function="ADCC"):
    # Re-Create Alter DataFrame with leftout subjects
    df_merged = importAlterDF()

    fullsubj = np.array(df_merged["subject"])  # Subjects only included in Alter
    leftout = []
    subjects, _, _ = getAxes()
    for index, i in enumerate(subjects):
        if i not in fullsubj:
            leftout.append((index, i))  # Subjects left out of Alter
    indices = [i[0] for i in leftout]

    func, _ = importFunction()
    df = pd.DataFrame(tensorFac[1][0])  # subjects x components matrix
    df = df.join(func, how="inner")
    df = df.iloc[indices]
    df = df.dropna()
    df_func = df[functions]
    df_variables = df.drop(["subject"] + functions, axis=1)

    X = df_variables
    Y = df_func[function]
    Y_pred = cross_val_predict(SVR(), X, Y, cv=10)
    accuracy = np.sqrt(r2_score(Y, Y_pred))
    print(f"Accuracy: {accuracy}")

    return Y, Y_pred
