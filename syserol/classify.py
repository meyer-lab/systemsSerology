""" Regression methods using Factorized Data. """
from sklearn.metrics import accuracy_score
from .dataImport import load_file, importAlterDF, selectAlter
from .regression import RegressionHelper


def getClassPred(X, df):
    """ Extract Ys for classification. """
    Y1 = (df["class.cp"] == "controller").astype(int)  # control 1, progress 0
    Y2 = (df["class.nv"] == "viremic").astype(int)  # viremic 1, nonviremic 0

    return RegressionHelper(X, Y1, classify=True), RegressionHelper(X, Y2, classify=True), Y1, Y2


def class_predictions(X):
    """ Predict subject class by progression (EC/VC vs TP/UP) or by viremia (EC/TP vs VC/UP) with Decomposed Tensor Data """
    # Load Data
    cp, nv, Y_cp, Y_nv = getClassPred(X, load_file("meta-subjects"))

    accuracies = {}
    accuracies["cp_all"] = accuracy_score(*selectAlter(Y_cp, cp[0]))
    accuracies["nv_all"] = accuracy_score(*selectAlter(Y_nv, nv[0]))
    return accuracies, cp[1], nv[1]

def four_class_predictions(X, random=False):
    """ Predict subject class for EC, VC, TP, or UP, with Decomposed Tensor Data """
    df = load_file("meta-subjects")
    Y = (df["class.etuv"]).replace(to_replace=["EC", "TP", "VC", "UP"], value = [1, 2, 3, 4])
    if random:
        X = X.sample(frac=1)
    Y_pred, _ = RegressionHelper(X, Y, classify=True, four=True)
    return accuracy_score(*selectAlter(Y, Y_pred))

def two_way_classifications():
    """ Predict classifications of subjects by progression (EC/VC vs TP/UP) or by viremia (EC/TP vs VC/UP) - Alter methods"""
    df = importAlterDF(function=False, subjects=True)

    # Subset, Z score
    X = df.drop(["subject", "class.etuv", "class.cp", "class.nv"], axis=1)
    cp, nv, Y1, Y2 = getClassPred(X, df)
    return accuracy_score(Y1, cp[0]), accuracy_score(Y2, nv[0])

def four_way_classification():
    """ Predict classifications of subjects - EC, VC, TP, or UP, according to Alter methods"""
    df = importAlterDF(function=False, subjects=True)
    X = df.drop(["subject", "class.etuv", "class.cp", "class.nv"], axis=1)
    Y = (df["class.etuv"]).replace(to_replace=["EC", "TP", "VC", "UP"], value = [1, 2, 3, 4])
    Y_pred, _ = RegressionHelper(X, Y, classify=True, four=True)

    return accuracy_score(Y, Y_pred)