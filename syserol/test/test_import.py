"""
Unit test file.
"""
import numpy as np
from ..dataImport import load_file, importLuminex, createCube
from ..COVID import pbsSubtractOriginal, Tensor3D


def test_files():
    """ Test that files are successfully loaded. """
    load_file("data-luminex")
    importLuminex()
    createCube()


def test_COVID_import():
    """ Test COVID import functions. """
    pbsSubtractOriginal()
    tensor, _ = Tensor3D()

    assert np.all(np.isfinite(tensor))
