""" Regression methods using Factorized Data. """
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from glmnet_python import cvglmnet, cvglmnetCoef
from .dataImport import load_file, importAlterDF, selectAlter


def getClassPred(X, df):
    """ Extract Ys for classification. """
    Y1 = (df["class.cp"] == "controller").astype(float).to_numpy()  # controllers are 1s, progressors are 0s
    Y2 = (df["class.nv"] == "viremic").astype(float).to_numpy()  # viremic = 1, nonviremic = 0

    return ClassifyHelper(X, Y1), ClassifyHelper(X, Y2), Y1, Y2


def class_predictions(X, evaluation="all"):
    """ Predict Subject Class with Decomposed Tensor Data """
    # Load Data
    cp, nv, Y_cp, Y_nv = getClassPred(X, load_file("meta-subjects"))

    cp_acc = accuracy_score(*selectAlter(Y_cp, cp[0], evaluation))
    nv_acc = accuracy_score(*selectAlter(Y_nv, nv[0], evaluation))
    return cp_acc, nv_acc, cp[1], nv[1]


def two_way_classifications():
    """ Predict classifications of subjects by progression (EC/VC vs TP/UP) or by viremia (EC/TP vs VC/UP) - Alter methods"""
    df = importAlterDF(function=False, subjects=True)

    # Subset, Z score
    X = df.drop(["subject", "class.etuv", "class.cp", "class.nv"], axis=1)
    cp, nv, Y1, Y2 = getClassPred(X, df)
    return accuracy_score(Y1, cp[0]), accuracy_score(Y2, nv[0])


def ClassifyHelper(X, Y):
    """ Function with common Logistic regression methods. """
    scores = []
    # separate dataframes
    X = scale(X)
    cvfit = cvglmnet(x=X.copy(), y=Y.copy(), nfolds=10, alpha=.8, keep=True, family='binomial', ptype='class', standardize=False)
    foldid = cvfit["foldid"]
    reps = 100
    for _ in range(reps):
        random.shuffle(foldid)
        cvfit = cvglmnet(x=X.copy(), y=Y.copy(), nfolds=10, alpha=.8, keep=True, foldid=foldid, family='binomial', ptype='class', standardize=False)
        Y_pred = cvfit['fit_preval'][:, np.where(cvfit["lambdau"] == cvfit['lambda_1se'])[0][0]]
        Y_pred = [int(x + 1) if x > .5 else int(x) for x in Y_pred]
        scores.append([accuracy_score(Y, Y_pred), Y_pred, cvfit])

    # TODO: Note that the accuracy on cross-validation is slightly lower than what glmnet returns.
    # score vs. accuracy_score(Y, Y_pred)
    return np.array(sorted(scores)[reps//2][1]), cvglmnetCoef(sorted(scores)[reps//2][2])[1:]