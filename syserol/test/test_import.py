"""
Unit test file.
"""
import pytest
from ..dataImport import load_file, importLuminex, createCube


@pytest.mark.skip()
def test_files():
    """ Test that files are successfully loaded. """
    load_file("data-luminex")
    importLuminex()
    createCube()
