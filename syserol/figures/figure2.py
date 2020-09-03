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
    for ii, func in enumerate(functions):
        _, _, acc = function_elastic_net(func)  # Alter Function Predictions
        accuracies[ii] = acc  # store accuracies
    for i, func in enumerate(functions):
        # our prediction accuracies
        _, _, accuracy = function_prediction(tensorFac, function=func, evaluation="Alter")
        accuracies[i + 6] = accuracy  # store

    # Create DataFrame
    model = ["Alter Model"] * 6 + ["Our Model"] * 6
    function = functions + functions
    data = {"Accuracy": accuracies, "Model": model, "Function": function}
    functions_df = pd.DataFrame(data)  # Function Prediction DataFrame, Figure 2B

    # Subjects left out of Alter
    preds = np.empty([83, 12])
    preds.fill(np.nan)
    for i, func in enumerate(functions):
        Y, Y_pred, _ = function_prediction(tensorFac, function=func, evaluation="notAlter")
        preds[0:len(Y), i] = Y
        preds[0:len(Y_pred), i + 6] = Y_pred

    df = pd.DataFrame(
        preds,
        columns=functions + functions,
    )
    X = pd.melt(df.iloc[:, 0:6])
    Y = pd.melt(df.iloc[:, 6:12])
    X.columns = ["Function", "Value"]
    Y.columns = ["Function", "Value"]
    subjects_out = pd.concat([X, Y], axis=1)
    subjects_out.columns = ["Function", "Value_x", "Function1", "Value_y"]
    subjects_out = subjects_out.drop(columns=["Function1"])  # DataFrame for Figure 2D

    # Gather Class Prediction Accuracies
    accuracyCvP, accuracyVvN, _, _ = two_way_classifications()  # Alter accuracies
    # Run our model
    subjects_matrix = tensorFac[1][0]
    cp_accuracy, nv_accuracy = class_predictions(subjects_matrix, False)  # Our accuracies

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
    a.tick_params(axis="x", labelsize=10)
    a.set_ylabel("Accuracy", fontsize=12)
    a.set_xlabel("Function", fontsize=12)

    # Class Plot
    b = sns.scatterplot(
        y="Accuracies", x="Class", style="Model", hue="Model", data=classes, ax=ax[1]
    )
    # Formatting
    b.plot([-0.5, 5.5], [avg, avg], "--", color="green")
    b.axvspan(-0.5, 0.5, alpha=0.1, color="grey")
    b.set_xlim(-0.5, 1.5)
    b.set_ylim(0.4, 1)
    b.grid(False)
    b.xaxis.tick_top()
    b.xaxis.set_label_position("top")
    b.set_ylabel("Accuracy", fontsize=12)
    b.set_xlabel("Class Prediction", fontsize=12)
    b.tick_params(axis="x", labelsize=10)

    # Function Predictions for Values left out of Alter Plot
    c = sns.scatterplot(
        x="Value_x", y="Value_y", hue="Function", data=subjects_out, ax=ax[2]
    )
    c.set_ylabel("Predicted Values", fontsize=12)
    c.set_xlabel("Actual Values", fontsize=12)

    subplotLabel(ax)

    return f
