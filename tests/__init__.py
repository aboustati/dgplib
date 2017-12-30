import unittest

from . import test_layer
from . import test_models

def test_module_suite():
    loader = unittest.TestLoader()
    #suite = loader.loadTestsFromModule(test_layer)
    suite = loader.discover('./')
    return suite
