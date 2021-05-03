"""
This creates Paper Figure 4.
"""

import numpy as np
import pandas as pd
import seaborn as sns
from pandas import concat
from ..regression import function_prediction, make_regression_df
from ..classify import class_predictions, class_predictions_df
from .common import subplotLabel, getSetup
from ..tensor import perform_CMTF
from ..dataImport import functions


def makeFigure():
    """ Compare prediction accuracies """
    ax, f = getSetup((12, 3), (1, 4))
    sns.set()
    rep = 10

    # Accuracy with difference component numbers
    df_function = []
    df_class = []
    resample = False
    for _ in range(rep):
        for r in np.arange(1, 11):
            tFac = perform_CMTF(r=r)[1][0]

            # Functional prediction
            accuracies = [function_prediction(tFac, resample=resample, function=f)[2] for f in functions]
            data = {"Accuracy": accuracies, "Components": r, "Function": functions}
            df_function.append(pd.DataFrame(data))

            # Classification
            accuracy = class_predictions(tFac, resample=resample)[0]
            df_class.append(pd.DataFrame.from_dict({"Class": accuracy.keys(),
                                                    "Accuracy": accuracy.values(),
                                                    "Components": r}))
    df_function = pd.concat(df_function)
    df_class = pd.concat(df_class)

    aa = sns.pointplot(x="Components", y="Accuracy", data=df_function, ci="sd", style="Function", hue="Function",
                       ax=ax[0], join=False, dodge=True)

    for i in np.arange(-0.5, 9.5, 2):
        aa.axvspan(i, i + 1, alpha=0.1, color="grey")
    aa.set_ylim(-0.3, 1)
    aa.grid(False)
    aa.legend(fontsize=8, title="Function", title_fontsize=9)

    # Classification plot
    bb = sns.pointplot(x="Components", y="Accuracy", data=df_class, ci="sd", style="Class", hue="Class",
                       ax=ax[1], join=False, dodge=True)
    for i in np.arange(-0.5, 9.5, 2):
        bb.axvspan(i, i + 1, alpha=0.1, color="grey")
    bb.set_ylim(0.2, 1)
    bb.grid(False)
    bb.legend(fontsize=8, title="Class", title_fontsize=9)

    # Show Similarity in Prediction of Alter Model and Our Model
    # Decompose Cube
    tFac = perform_CMTF()

    # Function Prediction DataFrame
    functions_df = concat([make_regression_df(tFac[1][0]) for _ in range(rep)])

    # Class Predictions DataFrame
    classes = concat([class_predictions_df(tFac[1][0]) for _ in range(rep)])

    # Function Plot
    cc = sns.pointplot(x="Function", y="Accuracy", data=functions_df, ci="sd", style="Model", hue="Model",
                       ax=ax[2], join=False, dodge=True)
    # Formatting
    shades = [-0.5, 1.5, 3.5]
    for i in shades:
        cc.axvspan(i, i + 1, alpha=0.1, color="grey")
    cc.set_xlim(-0.5, 5.5)
    cc.set_ylim(-0.3, 1)
    cc.grid(False)
    cc.xaxis.tick_top()
    cc.xaxis.set_label_position("top")
    cc.tick_params(axis="x")
    cc.set_ylabel("Accuracy")
    cc.set_xlabel("Function")
    cc.get_legend().remove()

    # Class Plot
    dd = sns.pointplot(x="Class", y="Accuracies", data=classes, ci="sd", style="Model", hue="Model",
                       ax=ax[3], join=False, dodge=True)
    # Formatting
    dd.axvspan(-0.5, 0.5, alpha=0.1, color="grey")
    dd.axvspan(1.5, 2.5, alpha=0.1, color="grey")
    dd.set_xlim(-0.5, 2.5)
    dd.set_ylim(0.2, 1)
    dd.grid(False)
    dd.xaxis.tick_top()
    dd.xaxis.set_label_position("top")
    dd.set_ylabel("Accuracy")
    dd.set_xlabel("Class Prediction")
    dd.tick_params(axis="x")
    dd.legend(fontsize=8.5, title="Model", title_fontsize=10)

    subplotLabel(ax)

    return f
