""" Regression methods. """
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from functools import reduce 
from .dataImport import createCube, importFunction, importLuminex, importGlycan, importIGG
from .tensor import perform_decomposition


def patientComponents(nComp=1):
    """ Generate factorization on cross-validation. """
    cube, _ = createCube()

    factors = perform_decomposition(cube, nComp)

    Y = importFunction()["ADCC"]

    idxx = np.isfinite(Y)
    Y = Y[idxx]
    X = factors[0][idxx, :]
    Y_pred = np.empty(Y.shape)

    Y_pred = cross_val_predict(ElasticNetCV(normalize=True), X, Y, cv=len(Y))

    model = ElasticNetCV(normalize=True).fit(X, Y)
    print(model.coef_)

    print(np.sqrt(r2_score(Y, Y_pred)))

    return Y, Y_pred

def function_elastic_net(function='ADCC'):
    #Import Luminex, Luminex-IGG, Function, and Glycan into DF
    df = importLuminex()
    lum = df.pivot(index='subject', columns = 'variable', values = 'value')
    glycan, df2 = importGlycan()
    glyc = df2.pivot(index='subject', columns = 'variable', values = 'value')
    func = importFunction()
    igg = importIGG()
    igg = igg.pivot(index='subject', columns = 'variable', values = 'value')
    data_frames = [lum, glyc, func, igg]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['subject'],
                                                how='inner'), data_frames)
    df_merged = df_merged.dropna()

    #separate dataframes
    df_func = df_merged[["ADCD", "ADCC", "ADNP", "CD107a", "IFNy", "MIP1b"]]
    df_variables = df_merged.drop(['subject','ADCD', 'ADCC', 'ADNP', 'CD107a', 'IFNy', 'MIP1b'], axis = 1)

    #perform regression
    Y = df_func[function]
    X = df_variables
    Y_pred = np.empty(Y.shape)

    regr = ElasticNetCV(normalize=True, max_iter = 10000)
    model = regr.fit(X, Y)
    Y_pred = cross_val_predict(ElasticNet(alpha = regr.alpha_, normalize = True, max_iter = 10000), X, Y, cv = 10)

    print(model.coef_)
    print(np.sqrt(r2_score(Y, Y_pred)))
    
    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))
