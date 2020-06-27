""" Regression methods. """
import numpy as np
import scipy as sp
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from .dataImport import createCube, importFunction
from .tensor import perform_decomposition, find_R2X


def patientComponents(nComp = 1):
    """ Generate factorization on cross-validation. """
    cube, _ = createCube()

    factors = perform_decomposition(cube, nComp)

    Y = importFunction()['ADCC']

    idxx = np.isfinite(Y)
    Y = Y[idxx]
    X = factors[0][idxx, :]
    Y_pred = np.empty(Y.shape)
    
    Y_pred = cross_val_predict(ElasticNetCV(normalize=True), X, Y, cv=len(Y))

    model = ElasticNetCV(normalize=True).fit(X, Y)
    print(model.coef_)

    print(np.sqrt(r2_score(Y, Y_pred)))
    
    return Y, Y_pred
