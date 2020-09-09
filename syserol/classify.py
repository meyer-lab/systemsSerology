""" Regression methods using Factorized Data. """
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from scipy.stats import zscore
from syserol.dataImport import (
    load_file,
    importAlterDF,
)


def getClassY(df):
    """ Extract Ys for classification. """
    Y1 = (df["class.cp"] == "controller").astype(int)  # controllers are 1s, progressors are 0s
    Y2 = (df["class.nv"] == "viremic").astype(int)  # viremic = 1, nonviremic = 0
    return Y1, Y2


def class_predictions(X, methodLR):
    """ Predict Subject Class with Decomposed Tensor Data """
    # Load Data
    cp, nv = getClassY(load_file("meta-subjects"))

    # Controller/Progressor classification
    _, cp_accuracy, _ = ClassifyHelper(X, cp, methodLR)

    # Viremic/Nonviremic classification
    _, nv_accuracy, _ = ClassifyHelper(X, nv, methodLR)

    return cp_accuracy, nv_accuracy


def two_way_classifications():
    """ Predict classifications of subjects by progression (EC/VC vs TP/UP) or by viremia (EC/TP vs VC/UP) - Alter methods"""
    df_merged = importAlterDF(function=False, subjects=True)

    # Subset, Z score
    X = df_merged.drop(
        ["subject", "class.etuv", "class.cp", "class.nv"], axis=1
    )
    X = X.apply(zscore)
    Y1, Y2 = getClassY(df_merged)

    # Predict Controller vs. Progressor
    _, accuracyCvP, confusionCvP = ClassifyHelper(X, Y1, True)

    # Predict Viremic vs. Nonviremic
    _, accuracyVvN, confusionVvN = ClassifyHelper(X, Y2, True)

    return accuracyCvP, accuracyVvN, confusionCvP, confusionVvN


def ClassifyHelper(X, Y, methodLR):
    """ Function with common Logistic regression methods. """
    if methodLR is True:
        regr = LogisticRegressionCV(n_jobs=-1, max_iter=1000)
        regr.fit(X, Y)

        if len(regr.coef_) < 50:
            print(f"Classification LR Coefficient: {regr.coef_}")

        clf = LogisticRegression(C=regr.C_[0], max_iter=1000)
    else:
        clf = SVC(kernel="rbf")

    Y_pred = cross_val_predict(clf, X, Y, cv=30, n_jobs=-1)
    confusion = confusion_matrix(Y, Y_pred)
    accuracy = accuracy_score(Y, Y_pred)
    return Y_pred, accuracy, confusion
