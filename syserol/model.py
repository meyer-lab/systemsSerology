""" Regression methods using Factorized Data. """
import pandas as pd
import numpy as np
from .tensor import impute, perform_decomposition
from .dataImport import createCube, importFunction
from sklearn.linear_model import ElasticNetCV, ElasticNet, LogisticRegressionCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

def function_predictions(function = 'ADCD'):
    cube, glyCube = createCube()
    factors = perform_decomposition(cube, 16)
    func = importFunction()
        
    df = pd.DataFrame(factors[0])
    df = df.join(func, how = 'inner')
    df = df.dropna()
    df_func = df[["ADCD", "ADCC", "ADNP", "CD107a", "IFNy", "MIP1b"]]
    df_variables = df.drop(['subject','ADCD', 'ADCC', 'ADNP', 'CD107a', 'IFNy', 'MIP1b'], axis = 1)

    X = df_variables
    Y = df_func[function]
    regr = ElasticNetCV(normalize=True, max_iter = 10000)
    model = regr.fit(X, Y)
    Y_pred = cross_val_predict(ElasticNet(alpha = regr.alpha_, normalize = True, max_iter = 10000), X, Y, cv = 10)

    print(model.coef_)
    print(np.sqrt(r2_score(Y, Y_pred)))
    
    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))
