""" Regression methods. """
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from .dataImport import importFunction, functions, importAlterDF, getAxes, AlterIndices


def function_elastic_net(function="ADCC"):
    """ Predict functions using elastic net according to Alter methods"""
    # Import Luminex, Luminex-IGG, Function, and Glycan into DF
    df = importAlterDF()
    func, _ = importFunction()
    df_merged = df.merge(func, on="subject", how="inner")
    # separate dataframes
    df_func = df_merged[functions]
    df_variables = df_merged.drop(["subject"] + functions, axis=1)

    # perform regression
    Y = df_func[function]
    X = df_variables
    Y_pred = np.empty(Y.shape)

    Y_pred = elastic_net_helper(X, Y)

    print(np.sqrt(r2_score(Y, Y_pred)))

    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))


def noCMTF_function_prediction(tensorFac, function="ADCC"):
    """ Predict functions using our decomposition and regression methods"""
    func, _ = importFunction()
    df = pd.DataFrame(tensorFac[1][0])  # subjects x components matrix
    df = df.join(func, how="inner")
    dropped = [index for index, row in df.iterrows() if row.isna().any()]
    df = df.dropna()
    df_func = df[functions]
    df_variables = df.drop(["subject"] + functions, axis=1)

    X = df_variables
    Y = df_func[function]
    Y_pred = elastic_net_helper(X, Y)

    accuracy = accuracy_alterSubjOnly(Y, Y_pred, dropped)

    return Y, Y_pred, accuracy


def SVR_noCMTF_function_prediction(tensorFac, function="ADCC"):
    """ Predict functions using our decomposition and SVR regression methods"""
    func, _ = importFunction()
    df = pd.DataFrame(tensorFac[1][0])  # subjects x components matrix
    df = df.join(func, how="inner")
    dropped = [index for index, row in df.iterrows() if row.isna().any()]
    df = df.dropna()
    df_func = df[functions]
    df_variables = df.drop(["subject"] + functions, axis=1)

    X = df_variables
    Y = df_func[function]
    Y_pred = cross_val_predict(SVR(), X, Y, cv=10)
    accuracy = accuracy_alterSubjOnly(Y, Y_pred, dropped)

    return Y, Y_pred, accuracy


def ourSubjects_function_prediction(tensorFac, function="ADCC"):
    """ Predict functions for subjects specifically left out of Alter using regression methods"""
    func, _ = importFunction()
    df = pd.DataFrame(tensorFac[1][0])  # subjects x components matrix
    df = df.join(func, how="inner")
    dropped = [index for index, row in df.iterrows() if row.isna().any()]
    df = df.dropna()
    df_func = df[functions]
    df_variables = df.drop(["subject"] + functions, axis=1)

    X = df_variables
    Y = df_func[function]
    Y_pred = elastic_net_helper(X, Y)

    Y_us, Yp_us, accuracy = accuracy_leftoutAlterSubj(Y, Y_pred, dropped)

    return Y_us, Yp_us


def SVR_ourSubjects_function_prediction(tensorFac, function="ADCC"):
    func, _ = importFunction()
    df = pd.DataFrame(tensorFac[1][0])  # subjects x components matrix
    df = df.join(func, how="inner")
    dropped = [index for index, row in df.iterrows() if row.isna().any()]
    df = df.dropna()
    df_func = df[functions]
    df_variables = df.drop(["subject"] + functions, axis=1)

    X = df_variables
    Y = df_func[function]
    Y_pred = cross_val_predict(SVR(), X, Y, cv=10)

    Y_us, Yp_us, accuracy = accuracy_leftoutAlterSubj(Y, Y_pred, dropped)

    return Y_us, Yp_us


def elastic_net_helper(X, Y):
    """ Helper Function for Elastic Net Regression"""
    regr = ElasticNetCV(normalize=True, max_iter=10000)
    regr.fit(X, Y)
    Y_pred = cross_val_predict(
        ElasticNet(alpha=regr.alpha_, normalize=True, max_iter=10000), X, Y, cv=10
    )
    return Y_pred


def accuracy_alterSubjOnly(Y, Ypred, dropped):
    """ Calculate the Accuracy for Only Subjects Included in Alter """
    indices = AlterIndices()

    Y_pred = Ypred.tolist()
    Y_ = Y.tolist()
    todrop = []
    for i in (dropped):  
        # Refill with NaN at Indices that were dropped before regression (Restore Ordering)
        Y_pred.insert(i, np.nan)
        Y_.insert(i, np.nan)
        if i in indices:  
            # Drop any subjects that were orginally in the dataframe, but not used for regression
            todrop.append(indices.index(i))
    indices_ = np.delete(
        indices, todrop
    )  # Indices that were included in Alter & the Regression

    # Subset Y and Y_pred to include only Alter+regressed subjects
    Yp_alter = np.array(Y_pred)[indices_]
    Y_alter = np.array(Y_)[indices_]
    print(f"Accuracy: {np.sqrt(r2_score(Y_alter, Yp_alter))}")

    return np.sqrt(r2_score(Y_alter, Yp_alter))


def accuracy_leftoutAlterSubj(Y, Ypred, dropped):
    """ Calculate accuracy for subjects that Alter could not predict and we did predict"""
    indices = AlterIndices()
    leftout = []  # Subjects leftout of Alter
    for i in np.arange(181):
        if i not in indices:
            leftout.append(i)

    Y_pred = Ypred.tolist()
    Y_ = Y.tolist()
    todrop = []
    for i in (dropped):  
        # Refill Ypred & Y with NaN at Indices that were dropped before regression (Restore Ordering)
        Y_pred.insert(i, np.nan)
        Y_.insert(i, np.nan)
        if i in leftout:
            todrop.append(leftout.index(i))
    leftout_indices = np.delete(
        leftout, todrop
    )  # Indices Leftout of Alter but also used in Regression

    # Subset Y and Y_pred to include non-Alter but regressed subjects
    Yp_us = np.array(Y_pred)[leftout_indices]
    Y_us = np.array(Y_)[leftout_indices]
    print(f"Accuracy: {np.sqrt(r2_score(Y_us, Yp_us))}")

    return Y_us, Yp_us, np.sqrt(r2_score(Y_us, Yp_us))
