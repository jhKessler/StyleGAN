import configure_pythonpath
from step_assert import StepAssert
import unittest

class TestStepAssert(unittest.TestCase):


    def test_step_assert(self):
        step_assert = StepAssert(tolerance=5)
        for i in range(5):
            self.assertTrue(step_assert(5))
        self.assertFalse(step_assert(5))
        step_assert.reset()
        
        for i in range(4):
            self.assertTrue(step_assert(5))
        self.assertTrue(step_assert(4))
        self.assertTrue(step_assert(5))