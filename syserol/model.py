""" Regression methods using Factorized Data. """
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV, ElasticNet, LogisticRegressionCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, confusion_matrix
from scipy.stats import zscore
from tensorly.kruskal_tensor import kruskal_to_tensor
from syserol.tensor import perform_CMTF
from syserol.dataImport import createCube, importFunction, load_file, importGlycan

def function_predictions(function = 'ADCD'):
    """ Predict Function using Factorized Antigen Data"""
    #import
    cube, glyCube = createCube()
    tensorFac, matrixFac, R2XX = perform_CMTF(cube, glyCube, 16)
    func, _ = importFunction()
    
    #arrange
    df = pd.DataFrame(tensorFac[1][0]) #subjects x components matrix
    df = df.join(func, how = 'inner')
    df = df.dropna()
    df_func = df[["ADCD", "ADCC", "ADNP", "CD107a", "IFNy", "MIP1b"]]
    df_variables = df.drop(['subject','ADCD', 'ADCC', 'ADNP', 'CD107a', 'IFNy', 'MIP1b'], axis = 1)

    #Regression
    X = df_variables
    Y = df_func[function]
    regr = ElasticNetCV(normalize=True, max_iter = 10000)
    model = regr.fit(X, Y)
    Y_pred = cross_val_predict(ElasticNet(alpha = regr.alpha_, normalize = True, max_iter = 10000), X, Y, cv = 10)

    print(model.coef_)
    print(np.sqrt(r2_score(Y, Y_pred)))

    return Y, Y_pred, np.sqrt(r2_score(Y, Y_pred))

def subject_predictions():
    """ Predict Subject Classifications using Factorized Antigen Data"""
    cube, glyCube = createCube()
    tensorFac, matrixFac, R2XX = perform_CMTF(cube, glyCube, 16)

    #Assemble data
    df = pd.DataFrame(tensorFac[1][0]) #subjects x components matrix
    subj = load_file('meta-subjects')
    df = df.join(subj, how = 'inner')
    df = df.dropna()

    #Subset, Z score
    df_class = df[["class.cp", "class.nv"]]
    df_variables = df.drop(['subject','class.etuv', 'class.cp', 'class.nv'], axis = 1)
    df_variables = df_variables.apply(zscore)

    #Predict Controller vs. Progressor
    Y1 = df_class['class.cp']
    Y1 = (Y1 == 'controller').astype(int) #controllers are 1s, progressors are 0s
    X1 = df_variables

    Y_pred1  = cross_val_predict(LogisticRegressionCV(max_iter = 10000), X1, Y1)
    model1 = LogisticRegressionCV().fit(X1, Y1)

    print(model1.coef_)
    accuracyCvP = confusion_matrix(Y1, Y_pred1)
    print(f"Confusion Matrix Controller vs. Progressor: {accuracyCvP} \n")

    #Predict Viremic vs. Nonviremic
    Y2 = df_class['class.nv']
    Y2 = (Y2 == 'viremic').astype(int) #viremic = 1, nonviremic = 0
    X2 = df_variables

    Y_pred2  = cross_val_predict(LogisticRegressionCV(max_iter = 10000), X2, Y2)
    model2 = LogisticRegressionCV().fit(X2, Y2)

    print(model2.coef_)
    accuracyVvN = confusion_matrix(Y2, Y_pred2)
    print(f"Confusion Matrix Viremic vs. Nonviremic: {accuracyVvN} \n")

    return accuracyCvP, accuracyVvN


def test_predictions(function = 'ADCD'):
    cube, glyCube = createCube()
    df, mapped = importFunction()
    glycan, dfGlycan = importGlycan()
    corr = list()

    for comp in np.arange(1, 16):
        tensorFac, matrixFac, R2XX = perform_CMTF(cube, glyCube, comp)
        reconMatrix = kruskal_to_tensor(matrixFac)
        x = mapped[function]
        j = len(glycan) + x
        orig = list()
        recon = list()
        for i in range(len(glyCube)):
            if (np.isfinite(glyCube[i,j])):
                orig.append(glyCube[i,j])
                recon.append(reconMatrix[i,j])
        corr.append(np.sqrt(r2_score(orig, recon)))
        print(f"Correlation for component {comp}: {np.sqrt(r2_score(orig, recon))}")

    return corr
