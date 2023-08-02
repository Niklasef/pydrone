import numpy as np
import unittest
from Block import *


# Place your function implementations here

class TestBlockFunctions(unittest.TestCase):

    def setUp(self):
        # Create a sample block for testing
        self.block = create_block(length=4, width=3, height=2)
        self.mass = 10.0

    def test_inertia(self):
        # Calculate the expected moment of inertia
        expected_I = 31.66666667

        # Test the inertia function
        self.assertAlmostEqual(inertia(self.block, self.mass), expected_I)

if __name__ == '__main__':
    unittest.main()
