""" Regression methods using Factorized Data. """
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score
from glmnet import LogitNet
from .dataImport import load_file, importAlterDF, selectAlter


def getClassPred(X, df):
    """ Extract Ys for classification. """
    Y1 = (df["class.cp"] == "controller").astype(int)  # controllers are 1s, progressors are 0s
    Y2 = (df["class.nv"] == "viremic").astype(int)  # viremic = 1, nonviremic = 0

    return ClassifyHelper(X, Y1), ClassifyHelper(X, Y2), Y1, Y2


def class_predictions(X, evaluation="all"):
    """ Predict Subject Class with Decomposed Tensor Data """
    # Load Data
    cp, nv, Y_cp, Y_nv = getClassPred(X, load_file("meta-subjects"))

    cp_acc = accuracy_score(*selectAlter(Y_cp, cp[0], evaluation))
    nv_acc = accuracy_score(*selectAlter(Y_nv, nv[0], evaluation))
    return cp_acc, nv_acc, cp[2], nv[2]


def two_way_classifications():
    """ Predict classifications of subjects by progression (EC/VC vs TP/UP) or by viremia (EC/TP vs VC/UP) - Alter methods"""
    df = importAlterDF(function=False, subjects=True)

    # Subset, Z score
    X = df.drop(["subject", "class.etuv", "class.cp", "class.nv"], axis=1)
    cp, nv, _, _ = getClassPred(X, df)
    return cp[1], nv[1]


def ClassifyHelper(X, Y):
    """ Function with common Logistic regression methods. """
    glmnet = LogitNet(alpha=.8, n_splits=10).fit(X, Y)
    Y_pred = cross_val_predict(glmnet, X, Y, cv=StratifiedKFold(n_splits=20), n_jobs=-1)

    return Y_pred, accuracy_score(Y, Y_pred), glmnet.coef_
