"""
Unit test file.
"""
from ..dataImport import load_file, importLuminex, createCube


def test_files():
    """ Test that files are successfully loaded. """
    load_file("data-luminex")
    importLuminex()
    createCube()
