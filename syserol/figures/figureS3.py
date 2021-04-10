import numpy as np
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessClassifier
from syserol.tensor import perform_CMTF
from syserol.dataImport import load_file
from syserol.figures.common import getSetup, subplotLabel


def makeFigure():
    ax, f = getSetup((5, 5), (2, 2))

    tFac, _, _ = perform_CMTF()
    X = tFac.factors[0]

    df = load_file("meta-subjects")
    y = (df["class.nv"] == "viremic").astype(int)

    kern = ConstantKernel() * RBF(np.ones(2), (1e-2, 1e14))
    kern += WhiteKernel(noise_level_bounds=(0.001, 0.8))
    estG = GaussianProcessClassifier(kern, n_restarts_optimizer=40)

    make_decision_plot(ax[0], estG, X[:, 4:6], y)

    # Add subplot labels
    subplotLabel(ax)

    return f


def make_decision_plot(ax, classifier, X, y):
    """ Make one decision plot. Only works with 2D data. """
    xx = np.linspace(-1.05, 1.05, 100)
    xx, yy = np.meshgrid(xx, xx.T)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    classifier.fit(X, y)

    # View probabilities:
    probas = classifier.predict_proba(Xfull)

    ax.imshow(probas[:, 0].reshape((100, 100)), extent=(-1.05, 1.05, -1.05, 1.05), origin='lower')
    ax.scatter(X[:, 0], X[:, 1], marker='.', c=y, edgecolor='k')
