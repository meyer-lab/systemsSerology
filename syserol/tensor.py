"""
Tensor decomposition methods
"""
import pickle
from pathlib import Path
from os.path import join, dirname
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac
from statsmodels.multivariate.pca import PCA
from .cmtf import coupled_matrix_tensor_3d_factorization

path_here = dirname(dirname(__file__))


def load_cache(r):
    """ Return a requested data file. """
    path = Path(join(path_here, "syserol/data/cache/factors" + str(r) + ".p"))

    if path.exists():
        data = pickle.load(open(path, "rb"))
        return data

    return None


def calcR2X(tensorIn, matrixIn, tensorFac, matrixFac):
    """ Calculate the R2X of CMTF. """
    tensorErr = np.nanvar(tl.kruskal_to_tensor(tensorFac) - tensorIn)
    matrixErr = np.nanvar(tl.kruskal_to_tensor(matrixFac) - matrixIn)

    return 1.0 - (tensorErr + matrixErr) / (np.nanvar(tensorIn) + np.nanvar(matrixIn))


def perform_CMTF(tensorIn, matrixIn, r):
    """ Perform CMTF decomposition. """
    tensor = np.copy(tensorIn)
    mask = np.isfinite(tensor).astype(int)
    tensor[mask == 0] = 0.0

    matrix = np.copy(matrixIn)
    mask_matrix = np.isfinite(matrix).astype(int)
    matrix[mask_matrix == 0] = 0.0

    # Check for a cache and if it matches return the result
    cache = load_cache(r)
    if cache is not None:
        tensorFac, matrixFac, R2Xcache = cache

        try:
            R2XX = calcR2X(tensorIn, matrixIn, tensorFac, matrixFac)
        except:
            R2XX = -1

        if np.isclose(R2XX, R2Xcache):
            print("Cache hit.")
            return tensorFac, matrixFac, R2Xcache
        else:
            print("Cache miss. Performing factorization.")

    # Initialize by running PARAFAC on the 3D tensor
    tensorFac = parafac(
        tensor,
        r,
        mask=mask,
        orthogonalise=True,
        normalize_factors=False,
        n_iter_max=300,
        linesearch=True,
    )
    tensor = tensor * mask + tl.kruskal_to_tensor(tensorFac, mask=1 - mask)
    assert np.all(np.isfinite(tensor))

    # Now run CMTF
    matrixFac = coupled_matrix_tensor_3d_factorization(matrix, mask_matrix=mask_matrix, init=tensorFac)

    # Solve for factors on remaining glycosylation matrix variation
    matrixResid = matrixIn - tl.kruskal_to_tensor(matrixFac)

    # pc = PCA(matrixResid, ncomp = 4, missing = "fill-em", max_em_iter = 200)
    # TODO: Incorporate this factorization into the existing tensors

    R2XX = calcR2X(tensorIn, matrixIn, tensorFac, matrixFac)

    print("CMTF R2X: " + str(R2XX))

    return tensorFac, matrixFac, R2XX
