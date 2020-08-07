"""
This creates Figure 9, Paper Figure 2.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from syserol.regression import function_elastic_net, two_way_classifications
from syserol.dataImport import importFunction, createCube
from syserol.model import cross_validation, SVM_2class_predictions #Function_Prediction_10FoldCV
from sklearn.metrics import r2_score
from syserol.figures.common import subplotLabel, getSetup
from syserol.tensor import perform_CMTF

def makeFigure():
    """ Show Similarity in Prediction of Alter Model and Our Model"""
    # Gather Function Prediction Accuracies
    arry = cross_validation() # Our Function Predictions in an array
    _, mapped = importFunction()
    accuracies = np.zeros(12)
    for ii, func in enumerate(mapped):
        _, _, acc = function_elastic_net(func) # Alter Function Predictions
        accuracies[ii] = acc # store accuracies
    for i in np.arange(6):
        x = arry[0:181, i]
        y = arry[0:181, i+6]
        idx = np.isfinite(x)
        accuracies[i+6] = np.sqrt(r2_score(x[idx], y[idx])) # Calculate our accuracies & store
    
    # Create DataFrame
    model = np.array(["Alter Model", "Alter Model", "Alter Model", "Alter Model", "Alter Model", "Alter Model",
                  "Our Model", "Our Model", "Our Model", "Our Model", "Our Model", "Our Model"])
    function = np.array(["ADCD", "ADCC", "ADNP", "CD107a", "IFNy", "MIP1b", "ADCD", "ADCC", "ADNP", "CD107a", "IFNy", "MIP1b"])
    data = {"Accuracy":accuracies, "Model":model, "Function":function}
    functions = pd.DataFrame(data)
    
    
    # Gather Class Prediction Accuracies
    accuracyCvP, accuracyVvN, _, _ = two_way_classifications() # Alter accuracies
    # Run our model
    cube, glyCube = createCube()
    tensorFac, matrixFac, R2XX = perform_CMTF(cube, glyCube, 6)
    subjects_matrix = tensorFac[1][0]
    cp_accuracy, nv_accuracy = SVM_2class_predictions(subjects_matrix) # Our accuracies
    
    # Create DataFrame
    baselineNV = .5083 #datasetEV3/Fc.array/class.nv/lambda.min/score_details.txt "No information rate"
    baselineCP = 0.5304 #datasetEV3/Fc.array/class.cp/lambda.min/score_details.txt "No information rate"
    avg = np.mean([baselineNV, baselineCP])
    accuracies = np.array([accuracyCvP, cp_accuracy, baselineCP, accuracyVvN, nv_accuracy, baselineNV])
    category = np.array(["CP_Alt", "CP_us", "CP Baseline", "NV_Alt", "NV_us", "NV Baseline"])
    model = np.array(["Alter Model", "Our Model", "Baseline", "Alter Model", "Our Model", "Baseline"])
    data = {"Accuracies":accuracies, "Class":category, "Model":model}
    classes = pd.DataFrame(data)
    
    # PLOT DataFrames
    ax, f = getSetup((10, 7), (1, 2))
    sns.set()
    # Function Plot
    a = sns.scatterplot(y="Accuracy", x="Function", style="Model", hue="Model", data=functions, ax=ax[0])
    # Formatting
    a.set_title("ADCD      ADCC      ADNP     CD107a      IFNy      MIP1b")
    shades = [-0.5, 1.5, 3.5]
    for i in shades:
        a.axvspan(i, i+1, alpha=0.1, color='grey')
    a.set_xlim(-0.5, 5.5)
    a.grid(False)
    a.set_xticks([])
    a.set_xlabel("")
    a.set_ylabel("Accuracy", fontsize=12)

    # Class Plot
    b = sns.scatterplot(y="Accuracies", x="Class", style="Model", hue="Model", data=classes, ax=ax[1])
    # Formatting
    b.plot([-0.5, 5.5], [avg, avg], '--', color='green')
    b.axvspan(-0.5, 2.5, alpha=0.1, color='grey')
    b.set_xlim(-0.5, 5.5)
    b.set_title("Progression                 Viremia")
    b.grid(False)
    b.set_xticks([])
    b.set_xlabel("")
    b.set_ylabel("Accuracy", fontsize=12)

    subplotLabel(ax)

    return f
