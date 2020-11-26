""" Regression methods using Factorized Data. """
import numpy as np
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score
from glmnet import LogitNet
from .dataImport import load_file, importAlterDF, AlterIndices


def getClassPred(X, df):
    """ Extract Ys for classification. """
    Y1 = (df["class.cp"] == "controller").astype(int)  # controllers are 1s, progressors are 0s
    Y2 = (df["class.nv"] == "viremic").astype(int)  # viremic = 1, nonviremic = 0

    return ClassifyHelper(X, Y1), ClassifyHelper(X, Y2), Y1, Y2


def class_predictions(X, evaluation="all"):
    """ Predict Subject Class with Decomposed Tensor Data """
    # Load Data
    cp, nv, Y_cp, Y_nv = getClassPred(X, load_file("meta-subjects"))
    accuracies = []
    for classpred in [[cp, Y_cp], [nv, Y_nv]]:
        Y = classpred[1]
        Y_pred = classpred[0][0]
        idx = np.zeros(Y.shape, dtype=np.bool)
        idx[AlterIndices()] = 1
        if evaluation == "Alter":
            Y, Y_pred = Y[idx], Y_pred[idx]
        elif evaluation == "notAlter":
            Y, Y_pred = Y[~idx], Y_pred[~idx]
        elif evaluation != "all":
            raise ValueError("Bad evaluation selection.")
        assert Y.shape == Y_pred.shape
        accuracies.append(accuracy_score(Y, Y_pred))
    return accuracies[0], accuracies[1], cp[2], nv[2]


def two_way_classifications():
    """ Predict classifications of subjects by progression (EC/VC vs TP/UP) or by viremia (EC/TP vs VC/UP) - Alter methods"""
    df = importAlterDF(function=False, subjects=True)

    # Subset, Z score
    X = df.drop(["subject", "class.etuv", "class.cp", "class.nv"], axis=1)
    cp, nv, _, _ = getClassPred(X, df)
    return cp[1], nv[1]


def ClassifyHelper(X, Y):
    """ Function with common Logistic regression methods. """
    glmnet = LogitNet(alpha=.8, n_jobs=10, n_splits=10).fit(X, Y)
    score = glmnet.cv_mean_score_[glmnet.lambda_best_ == glmnet.lambda_path_][0]

    Y_pred = cross_val_predict(glmnet, X, Y, cv=StratifiedKFold(n_splits=10), n_jobs=-1)

    # TODO: Note that the accuracy on cross-validation is slightly lower than what glmnet returns.
    # score vs. accuracy_score(Y, Y_pred)
    return Y_pred, score, glmnet.coef_
