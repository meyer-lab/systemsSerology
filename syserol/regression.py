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


def elasticNetFunc(X, Y):
    """ Function with common elastic net methods. """
    if X.shape[1] < 50:
        enet = LinearRegression()
        enet.fit(X, Y)
        print(f"Elastic Net Coefficient: {enet.coef_}")
    else:
        regr = ElasticNetCV(normalize=True, max_iter=10000, cv=30, n_jobs=-1, l1_ratio=0.8)
        regr.fit(X, Y)
        enet = ElasticNet(alpha=regr.alpha_, l1_ratio=regr.l1_ratio_, normalize=True, max_iter=10000)

    
    return enet


def function_elastic_net(function="ADCC"):
    """ Predict functions using elastic net according to Alter methods"""
    # Import Luminex, Luminex-IGG, Function, and Glycan into DF
    df = importAlterDF(function=True, subjects=False)
    df_merged = df.dropna()
    # separate dataframes
    Y = df_merged[function]
    df_variables = df_merged.drop(["subject"] + functions, axis=1)

    # perform regression
    enet = elasticNetFunc(df_variables, Y)
    Y_pred = cross_val_predict(enet, df_variables, Y, cv=len(Y), n_jobs=-1)
    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))


def function_prediction(tensorFac, function="ADCC", evaluation="all"):
    """ Predict functions using our decomposition and regression methods"""
    func, _ = importFunction()
    indices = AlterIndices()

    Y = func[function]
    X = tensorFac[1][0]  # subjects x components matrix
    notAlter = np.delete(np.arange(181), indices)
    dropped = np.unique(np.concatenate((np.nonzero(np.isnan(Y.to_numpy()))[0], notAlter)))
    Y_notAlter = Y[dropped][np.isfinite(Y)]
    X_notAlter = X[Y_notAlter.index]
    Y_Alter = Y[indices][np.isfinite(Y)]
    X_Alter = X[Y_Alter.index]
    Y = Y[np.isfinite(Y)]
    X = X[Y.index]

    enet = elasticNetFunc(X_Alter, Y_Alter)


    if evaluation == "all":
        Y_pred = cross_val_predict(enet, X, Y, cv=len(Y), n_jobs=-1)
        return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))
    elif evaluation == "Alter":
        Y_pred = cross_val_predict(enet, X_Alter, Y_Alter, cv=len(Y_Alter), n_jobs=-1)
        return Y_Alter, Y_pred, np.sqrt(r2_score(Y_Alter, Y_pred))
    elif evaluation == "notAlter":
        Y_pred = cross_val_predict(enet, X_notAlter, Y_notAlter, cv=len(Y_notAlter), n_jobs=-1)
        return Y_notAlter, Y_pred, np.sqrt(r2_score(Y_notAlter, Y_pred))

    raise ValueError("Wrong selection for evaluation.")


