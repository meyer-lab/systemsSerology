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
    AlterIndices
)


def function_elastic_net(function="ADCC"):
    """ Predict functions using elastic net according to Alter methods"""
    # Import Luminex, Luminex-IGG, Function, and Glycan into DF
    df = importAlterDF()
    func, _ = importFunction()
    df_merged = df.merge(func, on="subject", how="inner")
    # separate dataframes
    df_func = df_merged[functions]
    df_variables = df_merged.drop(
        ["subject"] + functions, axis=1
    )

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
    df_variables = df.drop(
        ["subject"] + functions, axis=1
    )

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
    df_variables = df.drop(
        ["subject"] + functions, axis=1
    )

    X = df_variables
    Y = df_func[function]
    Y_pred = cross_val_predict(SVR(), X, Y, cv=10)
    accuracy = accuracy_alterSubjOnly(Y, Y_pred, dropped)

    return Y, Y_pred, accuracy

def ourSubjects_function_prediction(tensorFac, function="ADCC"):
    """ Predict functions for subjects specifically left out of Alter using regression methods"""
    # Re-Create Alter DataFrame with leftout subjects
    df_alt = importAlterDF()

    fullsubj = np.array(df_alt["subject"])  # Subjects only included in Alter
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
    Y_pred = elastic_net_helper(X, Y)

    return Y, Y_pred

def SVR_ourSubjects_function_prediction(tensorFac, function="ADCC"):
    # Re-Create Alter DataFrame with leftout subjects
    df_alt = importAlterDF()

    fullsubj = np.array(df_alt["subject"])  # Subjects only included in Alter
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
    for i in dropped: # Refill with NaN at Indices that were dropped before regression 
        Y_pred.insert(i, np.nan)
        Y_.insert(i, np.nan)
        if i in indices: # Drop any subjects that were orginally in the dataframe, but not used for regression
            todrop.append(indices.index(i))
    indices_ = np.delete(indices, todrop)

    # Subset Y and Y_pred to include only Alter+regressed subjects
    Yp_alter = np.array(Y_pred)[indices_]
    Y_alter = np.array(Y_)[indices_]
    print(f"Accuracy: {np.sqrt(r2_score(Y_alter, Yp_alter))}")

    return np.sqrt(r2_score(Y_alter, Yp_alter))
