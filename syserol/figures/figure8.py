"""
This creates Figure 8.
"""

import numpy as np
from syserol.figures.common import subplotLabel, getSetup
from syserol.dataImport import createCube
from syserol.model import SVM_2class_predictions
from syserol.tensor import perform_CMTF


def makeFigure():
    """ Plot CP and NV SVM.SVC Prediction Accuracy Scores as Components for Decomposition Increase"""
    ax, f = getSetup((10, 10), (2, 1))
    cube, glyCube = createCube()
    cp_list = np.zeros(10)
    nv_list = np.zeros(10)
    for i in np.arange(1, 11):
        tensorFac, _, _, _ = perform_CMTF(cube, glyCube, i)
        cp_accuracy, nv_accuracy = SVM_2class_predictions(tensorFac[1][0])
        cp_list[i - 1] = cp_accuracy
        nv_list[i - 1] = nv_accuracy

    # Plotting
    ax[0].plot(np.arange(1, 11), cp_list)
    ax[0].set_xlabel("Number of Components", fontsize=12)
    ax[0].set_ylabel("Accuracy Score", fontsize=12)
    ax[0].set_title(
        "Controller/Progressor Class Prediction Accuracy for Increasing Component Decompositions",
        fontsize=15,
    )
    ax[0].set_xlim(1, 10)

    ax[1].plot(np.arange(1, 11), nv_list)
    ax[1].set_xlabel("Number of Components", fontsize=12)
    ax[1].set_ylabel("Accuracy Score", fontsize=12)
    ax[1].set_title(
        "Viremic/Non-Viremic Class Prediction Accuracy for Increasing Component Decompositions",
        fontsize=15,
    )
    ax[1].set_xlim(1, 10)

    subplotLabel(ax)

    return f
