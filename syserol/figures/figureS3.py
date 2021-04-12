import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessClassifier
from syserol.tensor import perform_CMTF
from syserol.dataImport import load_file
from syserol.figures.common import getSetup, subplotLabel


def makeFigure():
    ax, f = getSetup((5, 5), (2, 2))

    tFac = perform_CMTF()
    X = tFac.factors[0]

    df = load_file("meta-subjects")
    Y1 = (df["class.cp"] == "controller").astype(int)  # control 1, progress 0
    Y2 = (df["class.nv"] == "viremic").astype(int)  # viremic 1, nonviremic 0

    kern = ConstantKernel() * RBF(np.ones(2), (1e-2, 1e14))
    kern += WhiteKernel(noise_level_bounds=(0.0001, 0.8))
    estG = GaussianProcessClassifier(kern)

    make_decision_plot(ax[0], estG, X[:, np.array([0, 5])], Y2)
    make_decision_plot(ax[1], estG, X[:, np.array([2, 4])], Y1)

    # Add subplot labels
    subplotLabel(ax)

    return f


def make_decision_plot(ax, classifier, X, y):
    """ Make one decision plot. Only works with 2D data. """
    xx = np.linspace(-1.05, 1.05, 100)
    xx, yy = np.meshgrid(xx, xx.T)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    # Fit and get model probabilities
    classifier.fit(X, y)
    probas = classifier.predict_proba(Xfull)
    probas = probas > 0.5

    ax.imshow(probas[:, 0].reshape((100, 100)), extent=(-1.05, 1.05, -1.05, 1.05), origin='lower')
    ax.scatter(X[:, 0], X[:, 1], marker='.', c=y, edgecolor='k')
