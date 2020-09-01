"""
Unit test file.
"""
import unittest
import numpy as np
from ..dataImport import createCube
from ..tensor import perform_CMTF


class TestModel(unittest.TestCase):
    """ Test Class for Tensor related work. """
    def setUp(self):
        self.cube, self.glyCube = createCube()

    def test_R2X(self):
        """ Test to ensure R2X for higher components is larger. """
        arr = []
        for i in range(1, 3):
            facT, facM, tensorR2X, _ = perform_CMTF(self.cube, self.glyCube, i)
            self.assertTrue(np.all(np.isfinite(facT.factors[0])))
            self.assertTrue(np.all(np.isfinite(facT.factors[1])))
            self.assertTrue(np.all(np.isfinite(facT.factors[2])))
            self.assertTrue(np.allclose(facT.factors[0], facM.factors[0]))
            arr.append(tensorR2X)

        for j in range(len(arr) - 1):
            self.assertTrue(arr[j] < arr[j + 1])

        # confirm R2X is >= 0 and <=1
        self.assertGreaterEqual(np.min(arr), 0)
        self.assertLessEqual(np.max(arr), 1)
