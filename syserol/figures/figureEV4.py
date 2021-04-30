import numpy as np
import pandas as pd
import seaborn as sns
from .common import subplotLabel, getSetup
from ..tensor import perform_CMTF
from ..dataImport import functions
from ..regression import function_prediction
from ..classify import class_predictions

def makeFigure():
    df_function = []
    df_class = []
    resample = False
    for _ in range(10):
        for r in np.arange(1, 11):
            tFac = perform_CMTF(r=r)[1][0]

            # Functional prediction
            accuracies = [function_prediction(tFac, resample=resample, function=f)[2] for f in functions]
            data = {"Accuracy": accuracies, "Components": r, "Function": functions}
            df_function.append(pd.DataFrame(data))

            ## Classification
            accuracy = class_predictions(tFac, resample=resample)[0]
            df_class.append(pd.DataFrame.from_dict({"Class": accuracy.keys(),
                                          "Accuracy": accuracy.values(),
                                          "Components": r}))
    df_function = pd.concat(df_function)
    df_class = pd.concat(df_class)

    ## Function plot
    ax, f = getSetup((6, 3), (1, 2))
    sns.set()
    a = sns.pointplot(x="Components", y="Accuracy", data=df_function, ci="sd", style="Function", hue="Function",
                      ax=ax[0], join=False, dodge=True)

    for i in np.arange(-0.5, 9.5, 2):
        a.axvspan(i, i + 1, alpha=0.1, color="grey")
    a.set_ylim(-0.3, 1)
    a.grid(False)
    a.legend(fontsize=8, title="Function", title_fontsize=9)

    ## Classification plot
    b = sns.pointplot(x="Components", y="Accuracy", data=df_class, ci="sd", style="Class", hue="Class",
                      ax=ax[1], join=False, dodge=True)
    for i in np.arange(-0.5, 9.5, 2):
        b.axvspan(i, i + 1, alpha=0.1, color="grey")
    b.set_ylim(0.2, 1)
    b.grid(False)
    b.legend(fontsize=8, title="Class", title_fontsize=9)

    subplotLabel(ax)
    return f
