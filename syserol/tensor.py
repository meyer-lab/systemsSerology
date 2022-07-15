"""
Tensor decomposition methods
"""
import tensorly as tl
from tensorpack import perform_CMTF as p_CMTF
from .dataImport import createCube


tl.set_backend('numpy')


def perform_CMTF(tOrig=None, mOrig=None, r=6):
    """ Perform CMTF decomposition. """
    if tOrig is None:
        tOrig, mOrig = createCube()

    return p_CMTF(tOrig, mOrig, r=r, tol=1e-5, maxiter=100, progress=True)
