"""
Unit test file.
"""
import unittest
import numpy as np
import tensorly as tl
from ..dataImport import createCube
from ..tensor import find_R2X, perform_decomposition, R2X, perform_CMTF


class TestModel(unittest.TestCase):
    """ Test Class for Tensor related work. """
    def setUp(self):
        self.cube, self.glyCube = createCube()

    def test_R2X(self):
        """ Test to ensure R2X for higher components is larger. """
        self.assertTrue(np.isfinite(R2X(self.cube, self.cube)))
        self.assertTrue(R2X(self.cube, self.cube) == 1.0)

        arr = []
        for i in range(1, 4):
            factors = perform_decomposition(self.cube, i, iter_max=100)
            arr.append(find_R2X(self.cube, factors))

        for j in range(len(arr) - 1):
            self.assertTrue(arr[j] < arr[j + 1])

        # confirm R2X is >= 0 and <=1
        self.assertGreaterEqual(tl.min(arr), 0)
        self.assertLessEqual(tl.max(arr), 1)

    def test_CMTF(self):
        """ Test combined matrix-tensor factorization. """
        facT, facM, _ = perform_CMTF(self.cube, self.glyCube, 4)

        self.assertTrue(np.all(np.isfinite(facT[0])))
        self.assertTrue(np.allclose(facT[1][0], facM[1][0]))
