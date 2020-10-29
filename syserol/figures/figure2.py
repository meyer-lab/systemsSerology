"""
This creates Paper Figure 2.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from ..regression import (
    function_elastic_net,
    function_prediction
)
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
    accuracies = accuracies + [function_prediction(tFac, function=f, evaluation="Alter")[2] for f in functions]

    # Subjects left out of Alter
    accuracies = accuracies + [function_prediction(tFac, function=f, evaluation="notAlter")[2] for f in functions]

    # Create DataFrame
    model = ["Alter Model"] * 6 + ["Our Model"] * 6 + ["Excluded Cases"] * 6
    function = functions + functions + functions
    data = {"Accuracy": accuracies, "Model": model, "Function": function}
    functions_df = pd.DataFrame(data)  # Function Prediction DataFrame, Figure 2B

    # Gather Class Prediction Accuracies
    accuracyCvP, accuracyVvN = two_way_classifications()  # Alter accuracies
    # Run our model
    cp_accuracy, nv_accuracy, cp_coef, nv_coef = class_predictions(tFac[1][0])  # Our accuracies

    # Create DataFrame
    baselineNV = 0.5083  # datasetEV3/Fc.array/class.nv/lambda.min/score_details.txt "No information rate"
    baselineCP = 0.5304  # datasetEV3/Fc.array/class.cp/lambda.min/score_details.txt "No information rate"
    avg = np.mean([baselineNV, baselineCP])
    accuracies = np.array(
        [accuracyCvP, cp_accuracy, baselineCP, accuracyVvN, nv_accuracy, baselineNV]
    )
    category = ["Progression"] * 3 + ["Viremia"] * 3
    model = ["Alter Model", "Our Model", "Baseline"] * 2
    data = {"Accuracies": accuracies, "Class": category, "Model": model}
    classes = pd.DataFrame(data)  # Class Predictions DataFrame, Figure 2C

    """Provide details about our model"""
    #Factor data
    #Collect function component weights from elastic net prediction
    function_coefs = [function_prediction(tFac, function=f, evaluation="all")[3] for f in functions]
    flat_func_coefs = [func_coef for func in function_coefs for func_coef in func]
    function = [fun for fun in functions for i in range(6)]
    components = [1, 2, 3, 4, 5, 6] * 6
    data = {"Weights": flat_func_coefs, "Function": function, "Component": components}
    function_df = pd.DataFrame(data)

    #Collect classification component weights
    components = [1, 2, 3, 4, 5, 6] * 2
    category = ["Progression"] * 6 + ["Viremia"] * 6
    data = {"Weights": [ele for arr in np.hstack([cp_coef, nv_coef]) for ele in arr], "Class": category, "Component": components}
    class_df = pd.DataFrame(data)

    # PLOT DataFrames
    ax, f = getSetup((9, 6), (2, 3))
    sns.set()
    # Function Plot
    a = sns.pointplot(
        y="Accuracy",
        x="Function",
        hue="Model",
        markers=["o", "x", "d"],
        join=False,
        data=functions_df,
        ax=ax[0],
    )
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

    # Class Plot
    b = sns.scatterplot(
        y="Accuracies", x="Class", style="Model", hue="Model", data=classes, ax=ax[1]
    )
    # Formatting
    b.plot([-0.5, 5.5], [avg, avg], "--", color="green")
    b.axvspan(-0.5, 0.5, alpha=0.1, color="grey")
    b.set_xlim(-0.5, 1.5)
    b.set_ylim(0.45, 1)
    b.grid(False)
    b.xaxis.tick_top()
    b.xaxis.set_label_position("top")
    b.set_ylabel("Accuracy")
    b.set_xlabel("Class Prediction")
    b.tick_params(axis="x")

    # Function Predictions for Values left out of Alter Plot
    c = sns.pointplot(
        x="Function", y="Accuracy", join=False, data=subjects_out, ax=ax[2]
    )

    # Formatting
    shades = [-0.5, 1.5, 3.5]
    for i in shades:
        c.axvspan(i, i + 1, alpha=0.1, color="grey")
    c.set_xlim(-0.5, 5.5)
    c.set_ylim(0, 1)
    c.grid(False)
    c.xaxis.tick_top()
    c.xaxis.set_label_position("top")
    c.tick_params(axis="x")
    c.set_ylabel("Accuracy")
    c.set_xlabel("Function")

    #Component Weights
    sns.set()
    a = sns.barplot(data=function_df, x="Component", y="Weights", hue="Function", ax=ax[3])
    b = sns.barplot(data=class_df, x="Component", y="Weights", hue="Class", ax=ax[4])

    subplotLabel(ax)

    return f
