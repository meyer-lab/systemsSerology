"""
This creates Paper Figure 2.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from ..regression import function_elastic_net, function_prediction
from ..dataImport import functions
from ..classify import class_predictions, two_way_classifications
from .common import subplotLabel, getSetup
from ..tensor import perform_CMTF


def makeFigure():
    """ Show Similarity in Prediction of Alter Model and Our Model"""
    # Decompose Cube
    tFac, _, _ = perform_CMTF()

    # Gather Function Prediction Accuracies
    accuracies = [function_elastic_net(f)[2] for f in functions]
    accuracies = accuracies + [function_prediction(tFac[1][0], function=f, evaluation="Alter")[2] for f in functions]

    # Subjects left out of Alter
    accuracies = accuracies + [function_prediction(tFac[1][0], function=f, evaluation="notAlter")[2] for f in functions]

    # Create DataFrame
    model = ["Alter Model"] * 6 + ["Our Model"] * 6 + ["Excluded Cases"] * 6
    function = functions + functions + functions
    data = {"Accuracy": accuracies, "Model": model, "Function": function}
    functions_df = pd.DataFrame(data)  # Function Prediction DataFrame, Figure 2A

    # Gather Class Prediction Accuracies
    accuracyCvP, accuracyVvN = two_way_classifications()  # Alter accuracies
    # Run our model
    cp_accuracy, nv_accuracy, _, _ = class_predictions(tFac[1][0], "Alter")  # Our accuracies
    cp_notAlter, nv_notAlter, _, _ = class_predictions(tFac[1][0], "notAlter")

    # Create DataFrame
    baselineNV = 0.5083  # datasetEV3/Fc.array/class.nv/lambda.min/score_details.txt "No information rate"
    baselineCP = 0.5304  # datasetEV3/Fc.array/class.cp/lambda.min/score_details.txt "No information rate"
    accuracies = np.array([accuracyCvP, cp_accuracy, cp_notAlter, baselineCP, accuracyVvN, nv_accuracy, nv_notAlter, baselineNV])
    category = ["Progression"] * 4 + ["Viremia"] * 4
    model = ["Alter Model", "Our Model", "Excluded Cases", "Baseline"] * 2
    data = {"Accuracies": accuracies, "Class": category, "Model": model}
    classes = pd.DataFrame(data)  # Class Predictions DataFrame, Figure 2B

    # PLOT DataFrames
    ax, f = getSetup((6, 3), (1, 2))
    sns.set()
    # Function Plot
    a = sns.pointplot(y="Accuracy", x="Function", hue="Model", markers=["o", "x", "d"], join=False, data=functions_df, ax=ax[0],)
    # Formatting
    shades = [-0.5, 1.5, 3.5]
    for i in shades:
        a.axvspan(i, i + 1, alpha=0.1, color="grey")
    a.set_xlim(-0.5, 5.5)
    a.set_ylim(0, 1)
    a.grid(False)
    a.xaxis.tick_top()
    a.xaxis.set_label_position("top")
    a.tick_params(axis="x")
    a.set_ylabel("Accuracy")
    a.set_xlabel("Function")
    a.get_legend().remove()

    # Class Plot
    b = sns.scatterplot(y="Accuracies", x="Class", style="Model", hue="Model", data=classes, ax=ax[1])
    # Formatting
    b.axvspan(-0.5, 0.5, alpha=0.1, color="grey")
    b.set_xlim(-0.5, 1.5)
    b.set_ylim(0.45, 1)
    b.grid(False)
    b.xaxis.tick_top()
    b.xaxis.set_label_position("top")
    b.set_ylabel("Accuracy")
    b.set_xlabel("Class Prediction")
    b.tick_params(axis="x")
    b.legend(fontsize=8.5, title="Model", title_fontsize=10)

    subplotLabel(ax)

    return f
