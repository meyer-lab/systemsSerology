""" Regression methods. """
from functools import reduce
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, ElasticNet, LogisticRegressionCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score
from scipy.stats import zscore
from .dataImport import (
    createCube,
    importFunction,
    importLuminex,
    importGlycan,
    importIGG,
    load_file,
    importAlterDF,
    getAxes,
)
from .tensor import perform_CMTF


def patientComponents(nComp=1):
    """ Generate factorization on cross-validation. """
    cube, glyCube = createCube()
    factors = perform_CMTF(cube, glyCube, nComp)
    Y, _ = importFunction()
    Y = Y["ADCC"]

    idxx = np.isfinite(Y)
    Y = Y[idxx]
    X = factors[0][idxx, :]
    Y_pred = np.empty(Y.shape)

    Y_pred = cross_val_predict(ElasticNetCV(normalize=True), X, Y, cv=len(Y))

    model = ElasticNetCV(normalize=True).fit(X, Y)
    print(model.coef_)

    print(np.sqrt(r2_score(Y, Y_pred)))

    return Y, Y_pred


def function_elastic_net(function="ADCC"):
    """ Predict functions using elastic net according to Alter methods"""
    # Import Luminex, Luminex-IGG, Function, and Glycan into DF
    df_merged = importAlterDF()

    # separate dataframes
    df_func = df_merged[["ADCD", "ADCC", "ADNP", "CD107a", "IFNy", "MIP1b"]]
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


def two_way_classifications():
    """ Predict classifications of subjects by progression (EC/VC vs TP/UP) or by viremia (EC/TP vs VC/UP) - Alter methods"""
    # Import Luminex, Luminex-IGG, Subject group pairs, and Glycan into DF
    df = importLuminex()
    lum = df.pivot(index="subject", columns="variable", values="value")
    subj = load_file("meta-subjects")
    igg = importIGG()

    igg = igg.pivot(index="subject", columns="variable", values="value")
    data_frames = [lum, subj, igg]
    df_merged = reduce(
        lambda left, right: pd.merge(left, right, on=["subject"], how="inner"),
        data_frames,
    )
    df_merged = df_merged.dropna()

    # Subset, Z score
    df_class = df_merged[["class.cp", "class.nv"]]
    df_variables = df_merged.drop(
        ["subject", "class.etuv", "class.cp", "class.nv"], axis=1
    )
    df_variables = df_variables.apply(zscore)

    # Predict Controller vs. Progressor
    Y1 = df_class["class.cp"]
    Y1 = (Y1 == "controller").astype(int)  # controllers are 1s, progressors are 0s
    X1 = df_variables

    Y_pred1 = cross_val_predict(LogisticRegressionCV(), X1, Y1)
    model1 = LogisticRegressionCV().fit(X1, Y1)

    print(model1.coef_)
    confusionCvP = confusion_matrix(Y1, Y_pred1)
    accuracyCvP = accuracy_score(Y1, Y_pred1)
    print(f"Confusion Matrix Controller vs. Progressor: {confusionCvP} \n")

    # Predict Viremic vs. Nonviremic
    Y2 = df_class["class.nv"]
    Y2 = (Y2 == "viremic").astype(int)  # viremic = 1, nonviremic = 0
    X2 = df_variables

    Y_pred2 = cross_val_predict(LogisticRegressionCV(), X2, Y2)
    model2 = LogisticRegressionCV().fit(X2, Y2)

    print(model2.coef_)
    confusionVvN = confusion_matrix(Y2, Y_pred2)
    accuracyVvN = accuracy_score(Y2, Y_pred2)
    print(f"Confusion Matrix Viremic vs. Nonviremic: {confusionVvN} \n")

    return accuracyCvP, accuracyVvN, confusionCvP, confusionVvN


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
