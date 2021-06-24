import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gmean
from sklearn.metrics import accuracy_score
from .common import getSetup, subplotLabel
from ..dataImport import importAlterDF
from ..regression import RegressionHelper


def makeFigure():
    """Get a list of the axis objects and create a figure"""
    # Get list of axis objects
    ax, f = getSetup((9, 3), (1, 2))

    df = importAlterDF(subjects=True)
    gp120 = df.loc[:, df.columns.str.contains("IgG") & df.columns.str.contains("gp120")]
    p24 = df.loc[:, df.columns.str.contains("IgG") & df.columns.str.contains("p24")]
    IgGs = ["IgG1", "IgG2", "IgG3", "IgG4", "IgG"]

    test = pd.DataFrame(columns=["Class", "IgG", "gp120/p24 Ratio", "Gp120", "P24", "Progression"])
    accuracies = pd.DataFrame(columns=["IgG", "Accuracy"])
    for ig in IgGs:
        # Get data for boxplot
        gp_id = gp120.loc[:, gp120.columns.str.contains(ig)].mean(axis=1)
        p24_id = p24.loc[:, p24.columns.str.contains(ig)].mean(axis=1)
        data = {"Class": df["class.etuv"], "IgG": [ig] * 181, "gp120/p24 Ratio": gp_id / p24_id, "Gp120": gp_id, "P24": p24_id, "Progression": df["class.cp"]}
        test = pd.concat([test, pd.DataFrame(data)])

        # Make predictions
        df_pred = test.loc[test["IgG"] == ig, :]
        X = df_pred[["Gp120", "P24"]]
        Y = (df_pred["Progression"] == "controller").astype(int)
        Y_pred, coef, X, Y = RegressionHelper(X, Y)
        acc = accuracy_score(Y, Y_pred)
        data = {"IgG": ig, "Accuracy": acc}
        accuracies = accuracies.append(data, ignore_index=True)

    # Clip values < 0
    test["gp120/p24 Ratio"] = test["gp120/p24 Ratio"].clip(0)

    # Plot
    sns.boxplot(x="IgG", y="gp120/p24 Ratio", hue="Class", data=test, palette="colorblind", ax=ax[0])
    ax[0].set_ylim(-1, 15)

    sns.pointplot(x="IgG", y="Accuracy", data=accuracies, join=False, ax=ax[1])
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Controller/Progressor Predictions")

    subplotLabel(ax)
    return f
