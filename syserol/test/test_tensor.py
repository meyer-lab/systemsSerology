"""
Unit test file.
"""
import numpy as np
from ..dataImport import createCube
from ..tensor import perform_CMTF


def test_R2X():
    """ Test to ensure R2X for higher components is larger. """
    cube, glyCube = createCube()
    arr = []
    for i in range(1, 3):
        facT, facM, tensorR2X, _ = perform_CMTF(cube, glyCube, i)
        assert np.all(np.isfinite(facT.factors[0]))
        assert np.all(np.isfinite(facT.factors[1]))
        assert np.all(np.isfinite(facT.factors[2]))
        assert np.allclose(facT.factors[0], facM.factors[0])
        arr.append(tensorR2X)
    for j in range(len(arr) - 1):
        assert arr[j] < arr[j + 1]
    # confirm R2X is >= 0 and <=1
    assert np.min(arr) >= 0
    assert np.max(arr) <= 1
