import seaborn as sns
from ..COVID import Tensor4D, dimensionLabel4D
from ..tensor import cp_decomp
from .common import getSetup, subplotLabel


def makeFigure():
    ax, f = getSetup((6, 7), (4, 1))
    tensor, _ = Tensor4D()
    weeklabels, receptorslabels, agLabels = dimensionLabel4D()

    tfac = cp_decomp(tensor, 6)

    components = [str(ii + 1) for ii in range(tfac.rank)]
    comp_plot(tfac.factors[0], components, False, "Subjects", ax[0])
    comp_plot(tfac.factors[1], components, agLabels, "Antigens", ax[1])
    comp_plot(tfac.factors[2], components, receptorslabels, "Receptors", ax[2])
    comp_plot(tfac.factors[3], components, weeklabels, "Weeks", ax[3])
    subplotLabel(ax)
    return f


def comp_plot(factors, xlabel, ylabel, plotLabel, ax):
    """ Creates heatmap plots for each input dimension by component. """
    sns.heatmap(factors, cmap="PiYG", center=0, xticklabels=xlabel, yticklabels=ylabel, ax=ax)
    ax.set_xlabel("Components")
    ax.set_ylabel(plotLabel)
