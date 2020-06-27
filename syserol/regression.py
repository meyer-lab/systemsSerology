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
    cube, glyCube = createCube()

    factors = perform_decomposition(cube, nComp)
    
    for ii in range(3):
        factors_new = perform_decomposition(cube, nComp)
        print(find_R2X(cube, factors))
        
        if find_R2X(cube, factors_new) > find_R2X(cube, factors):
            print("Improved decomposition on second run.")
            factors = factors_new

    Y = importFunction()['ADCC']

    idxx = np.isfinite(Y)
    idxx = np.logical_and(idxx, np.all(np.isfinite(glyCube), axis=1))
    Y = Y[idxx]
    X = np.concatenate((factors[0][idxx, :], glyCube[idxx, :]), axis=1)
    Y = sp.stats.zscore(Y)
    Y_pred = np.empty(Y.shape)
    
    Y_pred = cross_val_predict(ElasticNetCV(normalize=True), X, Y, cv=len(Y))

    model = ElasticNetCV(normalize=True).fit(X, Y)
    print(model.coef_)

    print(np.sqrt(r2_score(Y, Y_pred)))
    
    return Y, Y_pred
