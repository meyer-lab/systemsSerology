"""
This creates Figure 6.
"""

from syserol.tensor import perform_CMTF
from syserol.dataImport import createCube
from .common import subplotLabel, getSetup
import numpy as np

def makeFigure():
    """ Plot elements contained in CMTF decomposed components by plotting Component vs. Component """
    cube, glyCube = createCube()
    tensorFac, matrixFac, _ = perform_CMTF(cube, glyCube, 10)
    
    size = 4
    ax, f = getSetup((10, 10), (4, 2))

    subjects = np.squeeze(tensorFac.factors[0])
    ax[0].plot(subjects[:, 0], subjects[:, 1], '.')
    ax[0].set_title("Tensor Fac: Subjects", fontsize=15)

    receptors = np.squeeze(tensorFac.factors[1])
    ax[1].plot(receptors[:, 0], receptors[:, 1], '.')
    ax[1].set_title("Tensor Fac: Receptors", fontsize=15)

    antigens = np.squeeze(tensorFac.factors[2])
    ax[2].plot(antigens[:, 0], antigens[:, 1], '.')
    ax[2].set_title("Tensor Fac: Antigens", fontsize=15)

    glyc = np.squeeze(matrixFac.factors[1])
    ax[3].plot(glyc[:, 0], glyc[:, 1], '.')
    ax[3].set_title("Matrix Fac: Glycans, Functions, and Classifications", fontsize=15)

    for i in range(size):
        ax[i].set_ylabel("Component 2")
        ax[i].set_xlabel("Component 1")
    
    # Components 3&4
    subjects = np.squeeze(tensorFac.factors[0])
    ax[4].plot(subjects[:, 2], subjects[:, 3], '.')
    ax[4].set_title("Tensor Fac: Subjects", fontsize=15)

    receptors = np.squeeze(tensorFac.factors[1])
    ax[5].plot(receptors[:, 2], receptors[:, 3], '.')
    ax[5].set_title("Tensor Fac: Receptors", fontsize=15)

    antigens = np.squeeze(tensorFac.factors[2])
    ax[6].plot(antigens[:, 2], antigens[:, 3], '.')
    ax[6].set_title("Tensor Fac: Antigens", fontsize=15)

    glyc = np.squeeze(matrixFac.factors[1])
    ax[7].plot(glyc[:, 2], glyc[:, 3], '.')
    ax[7].set_title("Matrix Fac: Glycans, Functions, and Classifications", fontsize=15)

    for i in np.arange(4,8):
        ax[i].set_ylabel("Component 4")
        ax[i].set_xlabel("Component 3")
    subplotLabel(ax)
    
    return f
