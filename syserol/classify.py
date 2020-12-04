""" Regression methods using Factorized Data. """
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
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
    glmnet = LogitNet(alpha=.8, n_jobs=10, n_splits=10).fit(X, Y)
    score = glmnet.cv_mean_score_[glmnet.lambda_best_ == glmnet.lambda_path_][0]

    Y_pred = cross_val_predict(glmnet, X, Y, cv=StratifiedKFold(n_splits=10), n_jobs=-1)

    # TODO: Note that the accuracy on cross-validation is slightly lower than what glmnet returns.
    # score vs. accuracy_score(Y, Y_pred)
    return Y_pred, score, glmnet.coef_


def ClassifySVC(X, Y):
    if X.shape[1] < 2:
        return np.zeros(Y.shape[0]), 0, np.zeros(X.shape[1])
    Cs = np.logspace(-4, 4, num=50)
    kernels = ['linear', 'rbf']
    skf = StratifiedKFold(n_splits=10)
    comp = X.shape[1]
    values_all = []
    for C in Cs:
        for kernel in kernels:
            for i in range(0, comp - 1):
                for j in range(i + 1, comp):
                    folds = []
                    Y_preds = []
                    for train, test in skf.split(X, Y):
                        double = np.vstack((X[train, i], X[train, j])).T
                        clf = SVC(kernel=kernel, C=C).fit(double, Y[train])
                        test_double = np.vstack((X[test, i], X[test, j])).T
                        Y_pred = clf.predict(test_double)
                        Y_preds.append(Y_pred)
                        score = accuracy_score(Y[test], Y_pred)
                        folds.append([score])
                    values_all.append([i, j, np.mean(folds), folds, Y_preds, kernel, C])
    df_comp = pd.DataFrame(values_all)
    df_comp.columns = ["First", "Second", "Score", "All_scores", "Y_pred", "kernel", "C"]
    df_comp = df_comp.sort_values(by="Score", ascending=False)
    components = np.zeros(comp)
    components[df_comp.iloc[0, 0]] = 1
    components[df_comp.iloc[0, 1]] = 1
    return df_comp.iloc[0, 4], df_comp.iloc[0, 2], components
