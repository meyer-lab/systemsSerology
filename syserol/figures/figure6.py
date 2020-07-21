"""
This creates Figure 6.
"""

from syserol.tensor import perform_CMTF
from syserol.dataImport import createCube
from .common import subplotLabel, getSetup
import numpy as np

def makeFigure():
    cube, glyCube = createCube()
    tensorFac, matrixFac, R2XX = perform_CMTF(cube, glyCube, 10)
    
    size = 5
    ax, f = getSetup((10, 10), (5, 2))

    subjects = np.squeeze(tensorFac.factors[0])
    ax[0].plot(subjects[:, 0], subjects[:, 1], '.')
    ax[0].set_title("Tensor Fac: Subjects", fontsize=15)

    receptors = np.squeeze(tensorFac.factors[1])
    ax[1].plot(receptors[:, 0], receptors[:, 1], '.')
    ax[1].set_title("Tensor Fac: Receptors", fontsize=15)

    antigens = np.squeeze(tensorFac.factors[2])
    ax[2].plot(antigens[:, 0], antigens[:, 1], '.')
    ax[2].set_title("Tensor Fac: Antigens", fontsize=15)

    subjects = np.squeeze(matrixFac.factors[0])
    ax[3].plot(subjects[:, 0], subjects[:, 1], '.')
    ax[3].set_title("Matrix Fac: Subjects", fontsize=15)

    glyc = np.squeeze(matrixFac.factors[1])
    ax[4].plot(glyc[:, 0], glyc[:, 1], '.')
    ax[4].set_title("Matrix Fac: Glycans, Functions, and Classifications", fontsize=15)

    for i in range(size):
        ax[i].set_ylabel("Component 2")
        ax[i].set_xlabel("Component 1")
    
    # Components 3&4
    subjects = np.squeeze(tensorFac.factors[0])
    ax[5].plot(subjects[:, 2], subjects[:, 3], '.')
    ax[5].set_title("Tensor Fac: Subjects", fontsize=15)

    receptors = np.squeeze(tensorFac.factors[1])
    ax[6].plot(receptors[:, 2], receptors[:, 3], '.')
    ax[6].set_title("Tensor Fac: Receptors", fontsize=15)

    antigens = np.squeeze(tensorFac.factors[2])
    ax[7].plot(antigens[:, 2], antigens[:, 3], '.')
    ax[7].set_title("Tensor Fac: Antigens", fontsize=15)

    subjects = np.squeeze(matrixFac.factors[0])
    ax[8].plot(subjects[:, 2], subjects[:, 3], '.')
    ax[8].set_title("Matrix Fac: Subjects", fontsize=15)

    glyc = np.squeeze(matrixFac.factors[1])
    ax[9].plot(glyc[:, 2], glyc[:, 3], '.')
    ax[9].set_title("Matrix Fac: Glycans, Functions, and Classifications", fontsize=15)

    for i in np.arange(5,10):
        ax[i].set_ylabel("Component 4")
        ax[i].set_xlabel("Component 3")
    subplotLabel(ax)
    
    return f
