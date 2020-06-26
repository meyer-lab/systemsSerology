"""
Unit test file.
"""
import unittest
from ..dataImport import load_file, importLuminex, createCube


class TestModel(unittest.TestCase):
    """ Test Class for Tensor related work. """

    def test_files(self):
        """ Test that files are successfully loaded. """
        load_file("data-luminex")
        importLuminex()
        createCube()
