"""
Unit test file.
"""
import numpy as np
from ..dataImport import load_file, importLuminex, createCube
from ..COVID import pbsSubtractOriginal, Tensor4D


def test_files():
    """ Test that files are successfully loaded. """
    load_file("data-luminex")
    importLuminex()
    createCube()


def test_COVID_import():
    """ Test COVID import functions. """
    pbsSubtractOriginal()
    tensor, subjects = Tensor4D()
    assert tensor.shape[0] == len(subjects)

    print(np.mean(np.isfinite(tensor), axis=(1, 2, 3)))
    print(np.mean(np.isfinite(tensor), axis=(0, 2, 3)))
    print(np.mean(np.isfinite(tensor), axis=(0, 1, 3)))
    print(np.mean(np.isfinite(tensor), axis=(0, 1, 2)))
