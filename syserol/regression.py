""" Regression methods. """
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
from .dataImport import createCube, importFunction
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

    Y_pred = cross_val_predict(ElasticNetCV(normalize=True), X, Y, cv=len(Y))

    model = ElasticNetCV(normalize=True).fit(X, Y)

    print(model.coef_)
    print(np.sqrt(r2_score(Y, Y_pred)))
    
    return np.sqrt(r2_score(Y, Y_pred))

def two_way_classifications():
    """Predict classifications of subjects by progression (EC/VC vs TP/UP) or by viremia (EC/TP vs VC/UP)"""
    
    #Import Luminex, Luminex-IGG, Subject group pairs, and Glycan into DF
    df = importLuminex()
    lum = df.pivot(index='subject', columns = 'variable', values = 'value')
    glycan, df2 = importGlycan()
    glyc = df2.pivot(index='subject', columns = 'variable', values = 'value')
    subj = load_file('meta-subjects')
    igg = importIGG()
    igg = igg.pivot(index='subject', columns = 'variable', values = 'value')
    data_frames = [lum, glyc, subj, igg]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['subject'],
                                                    how='inner'), data_frames)
    df_merged = df_merged.dropna()

    #Subset, Z score
    df_class = df_merged[["class.cp", "class.nv"]]
    df_variables = df_merged.drop(['subject','class.etuv', 'class.cp', 'class.nv'], axis = 1)
    df_variables = df_variables.apply(zscore) #Logistic Regression doesn't have normalization

    """Predict Controller vs. Progressor"""
    Y1 = df_class['class.cp']
    Y1 = (Y1 == 'controller').astype(int) #controllers are 1s, progressors are 0s 
    X1 = df_variables
    Y_pred1 = np.empty(Y1.shape)

    Y_pred1  = cross_val_predict(LogisticRegressionCV(), X1, Y1)
    model1 = LogisticRegressionCV().fit(X1, Y1)

    print(model1.coef_)
    accuracyCvP = np.sum(Y_pred1 == Y1)/len(Y1)
    print(f"Accuracy score Controller vs. Progressor: {accuracyCvP} \n")

    """Predict Viremic vs. Nonviremic"""
    Y2 = df_class['class.nv']
    Y2 = (Y2 == 'viremic').astype(int) #viremic = 1, nonviremic = 0
    X2 = df_variables 
    Y_pred2 = np.empty(Y2.shape)

    Y_pred2  = cross_val_predict(LogisticRegressionCV(), X2, Y2)
    model2 = LogisticRegressionCV().fit(X2, Y2)

    print(model2.coef_)
    accuracyVvN = np.sum(Y_pred2 == Y2)/len(Y2)
    print(f"Accuracy score Viremic vs. Nonviremic: {accuracyVvN} \n")
    
    return accuracyCvP, accuracyVvN

