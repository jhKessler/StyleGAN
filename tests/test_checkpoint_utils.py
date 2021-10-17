import unittest
import os
import configure_pythonpath
import checkpoint_utils

class TestUtils(unittest.TestCase):

    def test_id_is_valid(self):
        self.assertFalse(checkpoint_utils.id_is_valid("not an integer", test = True))
        self.assertFalse(checkpoint_utils.id_is_valid(1, test = True))
        open("tests/model_1.pkl", "a").close() # create file for testing
        self.assertTrue(checkpoint_utils.id_is_valid(1, test = True))
        os.remove("tests/model_1.pkl") # remove file

    def test_create_model_id(self):
        self.assertEquals(1, checkpoint_utils.create_model_id(test = True))
        open("tests/model_1.pkl", "a").close() # create file for testing
        self.assertEquals(2, checkpoint_utils.create_model_id(test = True))
        open("tests/model_3.pkl", "a").close() # create file for testing
        self.assertEquals(2, checkpoint_utils.create_model_id(test = True))
        os.remove("tests/model_1.pkl") # remove file
        os.remove("tests/model_3.pkl") # remove file