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
    tensorFac, _, _, _ = perform_CMTF()
    # Gather Function Prediction Accuracies
    accuracies = np.zeros(12)
    alter_function_preds = function_elastic_net()  # Alter Function Predictions
    for ii, result in enumerate(alter_function_preds):
        accuracies[ii] = result[0]  # store accuracies
    our_function_preds = function_prediction(tensorFac, evaluation="Alter")
    for i, func in enumerate(our_function_preds):
        # our prediction accuracies
        accuracies[i + ii] = func[0]  # store

    # Create DataFrame
    model = ["Alter Model"] * 6 + ["Our Model"] * 6
    function = functions + functions
    data = {"Accuracy": accuracies, "Model": model, "Function": function}
    functions_df = pd.DataFrame(data)  # Function Prediction DataFrame, Figure 2B

    # Subjects left out of Alter
    accuracies = np.zeros(6)
    not_Alter_function_preds = function_prediction(tensorFac, evaluation="notAlter")
    for i, func in enumerate(not_Alter_function_preds):
        accuracies[i] = func[0]
    # Create DataFrame
    data = {"Accuracy": accuracies, "Function": functions}
    subjects_out = pd.DataFrame(data) # DataFrame for Figure 2D

    # Gather Class Prediction Accuracies
    accuracyCvP, accuracyVvN = two_way_classifications()  # Alter accuracies
    # Run our model
    subjects_matrix = tensorFac[1][0]
    cp_accuracy, nv_accuracy = class_predictions(subjects_matrix)  # Our accuracies

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

    # PLOT DataFrames
    ax, f = getSetup((6, 5), (2, 2))
    sns.set()
    # Function Plot
    a = sns.pointplot(
        y="Accuracy",
        x="Function",
        hue="Model",
        markers=["o", "x"],
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

    subplotLabel(ax)

    return f
