#!/usr/bin/env python3

import seaborn as sns
import matplotlib
import numpy as np
import pickle
from syserol.dataImport import createCube, functions
from syserol.classify import SVM_2class_predictions
from syserol.regression import noCMTF_function_prediction
from syserol.tensor import perform_CMTF
from syserol.figures.common import subplotLabel, getSetup
from sklearn.metrics import r2_score

cube, glyCube = createCube()
# Find optimal component number based on prediction accuracies for function and class
# Change component number to see accuracy
tensorFac, _, _, _ = perform_CMTF(cube, glyCube, 6)
subjects_matrix = tensorFac[1][0]
cp_accuracy, nv_accuracy = SVM_2class_predictions(subjects_matrix) # class prediction accuracies for component i
    
# Function accuracies
func_accuracies = np.zeros(6)
r2s = np.zeros(6)
for i, func in enumerate(functions):
    _, _, accuracy = noCMTF_function_prediction(
            tensorFac, function=func
    )  # our prediction accuracies
    func_accuracies[i] = accuracy


print(f"Component Number = 6. CP Accuracy: {cp_accuracy}, NV Accuracy: {nv_accuracy}, Function Accuracies: {func_accuracies}")
