import unittest

from . import test_layer

def test_module_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_layer)
    return suite
