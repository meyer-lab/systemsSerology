"""
This creates Paper Figure 4.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import re
from pandas import concat
from ..regression import function_prediction, make_regression_df
from ..classify import class_predictions, class_predictions_df
from .common import subplotLabel, getSetup
from ..tensor import perform_CMTF
from ..dataImport import functions


def makeFigure():
    """ Compare prediction accuracies """
    ax, f = getSetup((9, 6), (2, 3))
    sns.set()
    rep = 10

    try:
        df_function = pd.read_csv('syserol/data/fig4_df_function.csv')
        df_class = pd.read_csv('syserol/data/fig4_df_class.csv')
        print("got df_function and df_class from csv's")
    except FileNotFoundError:
        print("could not get df_function and df_class from csv's",
              "calculating them from scratch....")

        # Accuracy with difference component numbers
        df_function = []
        df_class = []
        resample = False

        for r in np.arange(1, 15):
            tFac = perform_CMTF(r=r)[1][0]
            for _ in range(rep):
                # Functional prediction
                accuracies = [function_prediction(tFac, resample=resample, function=f)[
                    2] for f in functions]
                data = {"Accuracy": accuracies,
                        "Components": r, "Function": functions}
                df_function.append(pd.DataFrame(data))

                # Classification
                accuracy = class_predictions(tFac, resample=resample)[0]
                df_class.append(pd.DataFrame.from_dict({"Class": accuracy.keys(),
                                                        "Accuracy": accuracy.values(),
                                                        "Components": r}))
        df_function = pd.concat(df_function)
        df_class = pd.concat(df_class)

        df_function.to_csv('syserol/data/fig4_df_function.csv', index=False)
        df_class.to_csv('syserol/data/fig4_df_class.csv', index=False)
        print("done!")

    aa = sns.lineplot(x="Components", y="Accuracy", data=df_function, ci="sd", style="Function", hue="Function",
                      ax=ax[0])

    for i in np.arange(0.5, 14.5, 2):
        aa.axvspan(i, i + 1, alpha=0.1, color="grey")
    aa.set_ylim(0, 1)
    aa.set_xlim(0.5, 14.5)
    aa.grid(False)
    aa.legend(fontsize=8, title="Function", title_fontsize=9, handlelength=2)
    aa.set_xticks(range(1, 15))

    # Classification plot
    bb = sns.lineplot(x="Components", y="Accuracy", data=df_class, ci="sd", style="Class", hue="Class",
                      ax=ax[1], palette=sns.color_palette('magma', n_colors=3))
    for i in np.arange(0.5, 14.5, 2):
        bb.axvspan(i, i + 1, alpha=0.1, color="grey")
    bb.set_ylim(0.2, 1)
    bb.grid(False)
    bb.legend(fontsize=8, title="Class", title_fontsize=9, handlelength=2)
    bb.set_xticks(range(1, 15))
    bb.set_xlim(0.5, 14.5)

    # Show Similarity in Prediction of Alter Model and Our Model
    try:
        functions_df = pd.read_csv('syserol/data/fig4_functions_df.csv')
        classes = pd.read_csv('syserol/data/fig4_classes.csv')
        print("also got functions_df and classes from csv's")
    except FileNotFoundError:
        print("could not get functions_df and classes from csv's",
              "calculating them from scratch....")
        # Decompose Cube
        tFac = perform_CMTF()

        # Function Prediction DataFrame
        functions_df = concat([make_regression_df(tFac[1][0])
                              for _ in range(rep)])
        # Class Predictions DataFrame
        classes = concat([class_predictions_df(tFac[1][0])
                         for _ in range(rep)])
        functions_df.to_csv('syserol/data/fig4_functions_df.csv', index=False)
        classes.to_csv('syserol/data/fig4_classes.csv', index=False)
        print("done!")

    # Function Plot
    cc = sns.pointplot(x="Function", y="Accuracy", data=functions_df, ci="sd", style="Model", hue="Model",
                       ax=ax[3], join=False, dodge=True, markers=['o', 'X', 's'], hue_order=['CMTF', 'Alter et al', 'Randomized'])
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
    cc.legend(fontsize=8, title="Model", title_fontsize=9)

    # Class Plot
    dd = sns.pointplot(x="Class", y="Accuracies", data=classes, ci="sd", style="Model", hue="Model",
                       ax=ax[4], join=False, dodge=True, markers=['o', 'X', 's'], hue_order=['CMTF', 'Alter et al', 'Randomized'])
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
    dd.get_legend().remove()

    dd_labels = [re.sub('/', '/\n', x.get_text())
                 for x in dd.get_xticklabels()]

    dd.set_xticklabels(dd_labels)

    # Model prediction weight

    tFac = perform_CMTF()
    X = tFac.factors[0]
    ncomp = X.shape[1]
    nboot = 20

    class_df = pd.DataFrame()
    for _ in range(nboot):
        classes = []
        outt = class_predictions(X)
        classes.extend(outt[1])
        classes.extend(outt[2])

        data = {
            "Component Weight": classes,
            "Component": [str(x) for x in np.arange(1, ncomp + 1).tolist()] * 2,
            "Class": [x for i in [[j] * ncomp for j in ["Controller/Progressor", "Viremic/Non-Viremic"]] for x in i],
        }
        class_df = class_df.append(pd.DataFrame(data), ignore_index=True)

    funcs_df = pd.DataFrame()
    for _ in range(nboot):
        funcs = []
        for function in functions:
            coef = function_prediction(X, resample=True, function=function)[3]
            funcs.extend(coef)
        data = {
            "Component Weight": funcs,
            "Component": [str(x) for x in np.arange(1, ncomp + 1).tolist()] * 6,
            "Function": [x for i in [[j] * ncomp for j in functions] for x in i],
        }
        funcs_df = funcs_df.append(pd.DataFrame(data), ignore_index=True)

    sns.barplot(x="Component", y="Component Weight", ci="sd",
                hue="Function", data=funcs_df, errwidth=1, ax=ax[2])
    sns.barplot(x="Component", y="Component Weight", ci="sd", hue="Class", data=class_df,
                errwidth=2, ax=ax[5], palette=sns.color_palette('magma', n_colors=3))

    # Formatting
    shades = np.arange(-0.5, ncomp - 1, step=2.0)
    for axx in [ax[2], ax[5]]:
        for i in shades:
            axx.axvspan(i, i + 1, alpha=0.1, color="grey")
        axx.set_xlim(-0.5, ncomp - 0.5)

    ax[2].legend(fontsize=8, title="Function", title_fontsize=9)
    ax[5].legend(fontsize=8, title="Class", title_fontsize=9)

    # Add subplot labels
    subplotLabel([ax[0], ax[1], ax[3], ax[4], ax[2], ax[5]])

    return f
