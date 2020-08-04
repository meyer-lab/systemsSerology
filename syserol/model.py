""" Regression methods using Factorized Data. """
import numpy as np
import tensorly as tl
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNetCV, ElasticNet, LogisticRegressionCV
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from scipy.stats import zscore
from tensorly.kruskal_tensor import kruskal_to_tensor
from syserol.tensor import perform_CMTF
from syserol.dataImport import createCube, importFunction, importGlycan, load_file


def test_predictions(function="ADCD"):
    """ Test correlation between original glyCube matrix and CMTF decomposed/reconstructed matrix"""
    cube, glyCube = createCube()
    _, mapped = importFunction()
    glycan, _ = importGlycan()
    corr = list()

    for comp in np.arange(1, 16):
        _, matrixFac, _ = perform_CMTF(cube, glyCube, comp)
        reconMatrix = kruskal_to_tensor(matrixFac)
        x = mapped[function]
        j = len(glycan) + x
        orig = list()
        recon = list()
        for i in range(len(glyCube)):
            if np.isfinite(glyCube[i, j]):
                orig.append(glyCube[i, j])
                recon.append(reconMatrix[i, j])
        corr.append(np.sqrt(r2_score(orig, recon)))
        print(f"Correlation for component {comp}: {np.sqrt(r2_score(orig, recon))}")

    return corr


def Function_Prediction_10FoldCV(components=10):
    """ 10 Fold Cross Validation to Test Function Predictive Abilities"""
    cube, glyCube = createCube()
    _, mapped = importFunction()
    glycan, _ = importGlycan()

    X = glyCube
    matrix = np.zeros([181, 12])

    kf = KFold(n_splits=10, shuffle=True)  # split into 10 folds
    for train_index, test_index in kf.split(X):  # run cross validation
        copy = glyCube.copy()  # copy & restore original values at start of each cross validation fold
        matrix[test_index, 0:6] = copy[test_index, len(glycan) : len(glycan) + 6]  # store original value
        copy[test_index, len(glycan) : len(glycan) + 6] = np.nan  # artificially make the value NaN

        _, matrixFac, _ = perform_CMTF(cube, copy, components)  # run decomposition on new matrix
        pred_matrix = tl.kruskal_to_tensor(matrixFac)
        matrix[test_index, 6:13] = pred_matrix[test_index, len(glycan) : len(glycan) + 6]  # store predicted values

    return matrix

def SVM_2class_predictions(subjects_matrix):
    """ Predict Subject Class with Support Vector Machines and Decomposed Tensor Data"""
    # Load Data
    classes = load_file("meta-subjects")
    classes = classes.replace(to_replace=["controller", "progressor", "viremic", "nonviremic"], value=[1, 0, 1, 0])
    cp = np.array(classes["class.cp"])
    nv = np.array(classes["class.nv"])
    
    # Controller/Progressor classification
    X = subjects_matrix
    Y = cp
    # Kernel = RBF
    # Run SVM classifier model
    clf = SVC(kernel="rbf")
    y_pred1 = cross_val_predict(clf, X, Y, cv=10)
    cp_accuracy = accuracy_score(Y, y_pred1)
    
    # Viremic/Nonviremic classification 
    Y = nv
    # Kernel = RBF
    # Run SVM classifier model
    clf = SVC(kernel="rbf")
    y_pred2 = cross_val_predict(clf, X, Y, cv=10)
    nv_accuracy = accuracy_score(Y, y_pred2)
    
    return cp_accuracy, nv_accuracy
