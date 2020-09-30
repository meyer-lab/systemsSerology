""" Regression methods. """
import numpy as np
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
    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))


def function_prediction(tensorFac, function="ADCC", evaluation="all"):
    """ Predict functions using our decomposition and regression methods"""
    func, _ = importFunction()
    indices = AlterIndices()

    Y = func[function]
    X = tensorFac[1][0]  # subjects x components matrix

    # Split patients for various analysis methods
    notAlter = np.delete(
        np.arange(181), np.unique(np.concatenate((indices, np.where(np.isnan(Y))[0])))
    )
    dropped = np.unique(
        np.concatenate((np.nonzero(np.isnan(Y.to_numpy()))[0], notAlter))
    )
    Y_notAlter = Y[dropped][np.isfinite(Y)]
    Y_Alter = Y[indices][np.isfinite(Y)]

    # Perform Regression
    enet = LinearRegression().fit(X[Y_Alter.index], Y_Alter)
    print(f"LR Coefficient: {enet.coef_}")

    Y_pred_notAlter = enet.predict(X[Y_notAlter.index])
    Y_pred_Alter = cross_val_predict(enet, X[Y_Alter.index], Y_Alter, cv=len(Y_Alter), n_jobs=-1)

    if evaluation == "all":
        Y_pred = np.hstack((Y_pred_Alter, Y_pred_notAlter))
    elif evaluation == "Alter":
        Y, Y_pred = Y_Alter, Y_pred_Alter
    elif evaluation == "notAlter":
        Y, Y_pred = Y_notAlter, Y_pred_notAlter

    assert Y.shape == Y_pred.shape
    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))
