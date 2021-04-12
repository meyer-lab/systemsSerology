""" Regression methods using Factorized Data. """
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from .dataImport import load_file, importAlterDF, selectAlter
from .regression import RegressionHelper


def getClassPred(X, df, **kwargs):
    """ Extract Ys for classification. """
    Y1 = (df["class.cp"] == "controller").astype(int)  # control 1, progress 0
    Y2 = (df["class.nv"] == "viremic").astype(int)  # viremic 1, nonviremic 0
    Y3 = 2 * Y2 + Y1  # Split to all four classes

    returnList = []
    returnList.append(RegressionHelper(X, Y1, **kwargs))
    returnList.append(RegressionHelper(X, Y2, **kwargs))
    returnList.append(RegressionHelper(X, Y3, **kwargs))
    return returnList


def class_predictions(X, **kwargs):
    """ Predict Subject Class with Decomposed Tensor Data """
    # Load Data
    cp, nv, all = getClassPred(X, load_file("meta-subjects"), **kwargs)

    accuracies = {}
    accuracies["cp"] = accuracy_score(*selectAlter(cp[3], cp[0]))
    accuracies["nv"] = accuracy_score(*selectAlter(nv[3], nv[0]))
    accuracies["all"] = accuracy_score(*selectAlter(all[3], all[0]))
    return accuracies, cp[1], nv[1]


def class_predictions_df(X, resample=False):
    # Alter accuracies
    alterAcc = two_way_classifications(resample=resample)

    # Our accuracies
    accuracy = class_predictions(X, resample=resample)[0]

    # Our accuracies baseline
    shuffled = class_predictions(X, randomize=True)[0]

    # Create DataFrame
    accuracies = np.array(
        [
            alterAcc["cp"],
            accuracy["cp"],
            shuffled["cp"],
            alterAcc["nv"],
            accuracy["nv"],
            shuffled["nv"],
            alterAcc["all"],
            accuracy["all"],
            shuffled["all"],
        ]
    )
    category = ["Controller/Progressor"] * 3 + ["Viremic/Non-Viremic"] * 3 + ["Four-Class"] * 3
    model = ["CMTF", "Alter et al", "Randomized"] * 3
    data = {"Accuracies": accuracies, "Class": category, "Model": model}
    return pd.DataFrame(data)


def two_way_classifications(resample=False):
    """ Predict classifications of subjects by progression (EC/VC vs TP/UP) or by viremia (EC/TP vs VC/UP) - Alter methods"""
    df = importAlterDF(function=False, subjects=True)

    # Subset, Z score
    X = df.drop(["subject", "class.etuv", "class.cp", "class.nv"], axis=1)
    cp, nv, all = getClassPred(X, df, resample=resample)

    accuracies = {}
    accuracies["cp"] = accuracy_score(cp[3], cp[0])
    accuracies["nv"] = accuracy_score(nv[3], nv[0])
    accuracies["all"] = accuracy_score(all[3], all[0])
    return accuracies
