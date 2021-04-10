import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessClassifier
from syserol.tensor import perform_CMTF
from syserol.dataImport import load_file
from syserol.figures.common import getSetup, subplotLabel
from syserol.classify import class_predictions


def makeFigure():
    tFac, _, _ = perform_CMTF()
    X = tFac.factors[0]

    outt = class_predictions(X)
    df = load_file("meta-subjects")
    y = (df["class.nv"] == "viremic").astype(int)
    y_pred = outt[2][0]

    kern = ConstantKernel() * RBF(np.ones(2), (1e-2, 1e14))
    kern += WhiteKernel(noise_level_bounds=(0.001, 0.8))
    estG = GaussianProcessClassifier(kern, n_restarts_optimizer=40)

    classifiers = {
        "GPC-nv": estG
    }

    X = np.vstack([X[:, 4], X[:, 5]]).T
    compAmin, compAmax = X[:, 0].min() - X[:, 0].std(), X[:, 0].max() + X[:, 0].std()
    compBmin, compBmax = X[:, 1].min() - X[:, 1].std(), X[:, 1].max() + X[:, 1].std()

    xx = np.linspace(compAmin, compAmax, 100)
    yy = np.linspace(compBmin, compBmax, 100).T
    xx, yy = np.meshgrid(xx, yy)
    Xfull = np.c_[xx.ravel(), yy.ravel()]

    n_classifiers = len(classifiers)

    f = plt.figure(figsize=(3 * 2, n_classifiers * 2))
    plt.subplots_adjust(bottom=.1, top=.95)

    for index, (name, classifier) in enumerate(classifiers.items()):
        classifier.fit(X, y)

        # View probabilities:
        probas = classifier.predict_proba(Xfull)
        n_classes = np.unique(y_pred).size
        for k in range(n_classes):
            ax = plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
            plt.title("Class %d" % k)
            if k == 0:
                plt.ylabel(name)
            plt.imshow(probas[:, k].reshape((100, 100)),
                       extent=(compAmin, compAmax, compBmin, compBmax), origin='lower')
            plt.xticks(())
            plt.yticks(())
        
            plt.scatter(X[:, 0], X[:, 1], marker='.', c=y, edgecolor='k')

    # Add subplot labels
    subplotLabel(ax)

    return f
