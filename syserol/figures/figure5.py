"""
This creates Paper Figure 5.
"""

import seaborn as sns
from pandas import concat
from ..regression import make_regression_df
from ..classify import class_predictions_df
from .common import subplotLabel, getSetup
from ..tensor import perform_CMTF


def makeFigure():
    """ Show Similarity in Prediction of Alter Model and Our Model"""
    # Decompose Cube
    tFac = perform_CMTF()
    rep = 10

    # Function Prediction DataFrame, Figure 5A
    functions_df = concat([make_regression_df(tFac[1][0]) for _ in range(rep)])
    #functions_df = functions_df.groupby(['Model', 'Function'], as_index=False).agg({'Accuracy': ['mean', 'std']})
    #functions_df.columns = ['Model', 'Function', 'Accuracy', 'std']

    # Class Predictions DataFrame, Figure 5B
    classes = concat([class_predictions_df(tFac[1][0]) for _ in range(rep)])

    # PLOT DataFrames
    ax, f = getSetup((6, 3), (1, 2))
    sns.set()
    # Function Plot
    #a = sns.pointplot(y="Accuracy", x="Function", style="Model", hue="Model", data=functions_df, ax=ax[0], join=False)
    a = sns.pointplot(x="Function", y="Accuracy", data=functions_df, ci="sd", style="Model", hue="Model",
                      ax=ax[0], join=False, dodge=True)
    # Formatting
    shades = [-0.5, 1.5, 3.5]
    for i in shades:
        a.axvspan(i, i + 1, alpha=0.1, color="grey")
    a.set_xlim(-0.5, 5.5)
    a.set_ylim(-0.3, 1)
    a.grid(False)
    a.xaxis.tick_top()
    a.xaxis.set_label_position("top")
    a.tick_params(axis="x")
    a.set_ylabel("Accuracy")
    a.set_xlabel("Function")
    a.get_legend().remove()

    # Class Plot
    b = sns.pointplot(x="Class", y="Accuracies", data=classes, ci="sd", style="Model", hue="Model",
                      ax=ax[1], join=False, dodge=True)
    b = sns.scatterplot(y="Accuracies", x="Class", style="Model", hue="Model", data=classes, ax=ax[1])
    # Formatting
    b.axvspan(-0.5, 0.5, alpha=0.1, color="grey")
    b.axvspan(1.5, 2.5, alpha=0.1, color="grey")
    b.set_xlim(-0.5, 2.5)
    b.set_ylim(0.2, 1)
    b.grid(False)
    b.xaxis.tick_top()
    b.xaxis.set_label_position("top")
    b.set_ylabel("Accuracy")
    b.set_xlabel("Class Prediction")
    b.tick_params(axis="x")
    b.legend(fontsize=8.5, title="Model", title_fontsize=10)

    subplotLabel(ax)

    return f
