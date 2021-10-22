import unittest
import os
import configure_pythonpath
from clients import CheckpointClient
from args import Args

class TestCheckpointUtils(unittest.TestCase):


    def test_id_is_valid(self):
        self.assertFalse(CheckpointClient.id_is_valid("not an integer", test = True))
        self.assertFalse(CheckpointClient.id_is_valid(1, test = True))
        open("tests/model_1.pkl", "a").close() # create file for testing
        self.assertTrue(CheckpointClient.id_is_valid(1, test = True))
        os.remove("tests/model_1.pkl") # remove file


    def test_create_model_id(self):
        self.assertEquals(1, CheckpointClient.create_model_id(test = True))
        open("tests/model_1.pkl", "a").close() # create file for testing
        self.assertEquals(2, CheckpointClient.create_model_id(test = True))
        open("tests/model_3.pkl", "a").close() # create file for testing
        self.assertEquals(2, CheckpointClient.create_model_id(test = True))
        os.remove("tests/model_1.pkl") # remove file
        os.remove("tests/model_3.pkl") # remove file

    
    def test_create_new_model(self):
        # test creating new model
        test_args = Args
        Args.MODEL_ID = None
        Args.RESET_MODEL = True
        self.assertRaises(RuntimeError, CheckpointClient.get_checkpoint, Args)
        Args.RESET_MODEL = False
        new_checkpoint = CheckpointClient.get_checkpoint(Args)
        model_dict = new_checkpoint.get_dict()
        self.assertTrue(model_dict["step"] == model_dict["iteration"] == model_dict["samples"] == 0)
        self.assertTrue(model_dict["preview_noise"].shape == (5, Args.NUM_PROGRESS_IMGS, 2, Args.NOISE_DIM))
