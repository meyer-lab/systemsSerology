""" Regression methods. """
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, ElasticNet
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from .dataImport import (
    createCube,
    importFunction,
    load_file,
    functions,
    importAlterDF,
    getAxes,
)
from .tensor import perform_CMTF



def function_elastic_net(function="ADCC"):
    """ Predict functions using elastic net according to Alter methods"""
    # Import Luminex, Luminex-IGG, Function, and Glycan into DF
    df_merged = importAlterDF()

    # separate dataframes
    df_func = df_merged[functions]
    df_variables = df_merged.drop(
        ["subject", "ADCD", "ADCC", "ADNP", "CD107a", "IFNy", "MIP1b"], axis=1
    )

    # perform regression
    Y = df_func[function]
    X = df_variables
    Y_pred = np.empty(Y.shape)

    regr = ElasticNetCV(normalize=True, max_iter=10000)
    regr.fit(X, Y)
    Y_pred = cross_val_predict(
        ElasticNet(alpha=regr.alpha_, normalize=True, max_iter=10000), X, Y, cv=10
    )

    print(np.sqrt(r2_score(Y, Y_pred)))

    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))


def noCMTF_function_prediction(components=6, function="ADCC"):
    """ Predict functions using our decomposition and regression methods"""
    cube, glyCube = createCube()
    tensorFac, matrixFac, _ = perform_CMTF(cube, glyCube, components)

    func, _ = importFunction()
    df = pd.DataFrame(tensorFac[1][0])  # subjects x components matrix
    df = df.join(func, how="inner")
    df = df.dropna()
    df_func = df[["ADCD", "ADCC", "ADNP", "CD107a", "IFNy", "MIP1b"]]
    df_variables = df.drop(
        ["subject", "ADCD", "ADCC", "ADNP", "CD107a", "IFNy", "MIP1b"], axis=1
    )

    X = df_variables
    Y = df_func[function]
    regr = ElasticNetCV(normalize=True, max_iter=10000)
    model = regr.fit(X, Y)
    Y_pred = cross_val_predict(
        ElasticNet(alpha=regr.alpha_, normalize=True, max_iter=10000), X, Y, cv=10
    )

    print(f"Components: {components}, Accuracy: {np.sqrt(r2_score(Y, Y_pred))}")

    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))


def ourSubjects_function_prediction(components=6, function="ADCC"):
    """ Predict functions for subjects specifically left out of Alter using regression methods"""
    # Re-Create Alter DataFrame with leftout subjects
    df_merged = importAlterDF()

    fullsubj = np.array(df_merged["subject"])  # Subjects only included in Alter
    leftout = []
    subjects, _, _ = getAxes()
    for index, i in enumerate(subjects):
        if i not in fullsubj:
            leftout.append((index, i))  # Subjects left out of Alter
    indices = [i[0] for i in leftout]

    cube, glyCube = createCube()
    tensorFac, matrixFac, _ = perform_CMTF(cube, glyCube, components)

    func, _ = importFunction()
    df = pd.DataFrame(tensorFac[1][0])  # subjects x components matrix
    df = df.join(func, how="inner")
    df = df.iloc[indices]
    df = df.dropna()
    df_func = df[["ADCD", "ADCC", "ADNP", "CD107a", "IFNy", "MIP1b"]]
    df_variables = df.drop(
        ["subject", "ADCD", "ADCC", "ADNP", "CD107a", "IFNy", "MIP1b"], axis=1
    )

    X = df_variables
    Y = df_func[function]
    regr = ElasticNetCV(normalize=True, max_iter=10000)
    model = regr.fit(X, Y)
    Y_pred = cross_val_predict(
        ElasticNet(alpha=regr.alpha_, normalize=True, max_iter=10000), X, Y, cv=10
    )

    return Y, Y_pred
