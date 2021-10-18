import configure_pythonpath
from checkpoint import Checkpoint
import unittest
import os
from args import Args

class TestCheckpoint(unittest.TestCase):

    @staticmethod
    def return_test_dict() -> dict:
        """
        Returns dict with only zeroes for testing
        """
        test_args = Args
        test_args.MODEL_ID = 999
        test_dict = {
                "Args": test_args,
                "generator": 0,
                "g_optimizer": 0,
                "discriminator": 0,
                "d_optimizer": 0,
                "step": 0,
                "iteration": 0,
                "samples": 0,
                "start_time": 0,
                "preview_noise": 0
            }
        return test_dict
    

    def test_get_filepath(self):
        test_args = Args
        Args.MODEL_ID = 1
        self.assertEquals(Checkpoint.get_filepath(Args), "checkpoints/model_1.pkl")
        Args.MODEL_ID = 2
        self.assertEquals(Checkpoint.get_filepath(Args), "checkpoints/model_2.pkl")


    def test_create_checkpoint(self):
        # test that invalid dict results in error
        self.assertRaises(AttributeError, Checkpoint.create, dict())
        test_dict = TestCheckpoint.return_test_dict()
        checkpoint = Checkpoint.create(test_dict)
        cp_dict = checkpoint.get_dict()
        cp_dict.pop("Args")
        self.assertEquals(sum(cp_dict.values()), 0)

    
    def test_save_checkpoint(self):
        test_dict = TestCheckpoint.return_test_dict()
        checkpoint = Checkpoint(test_dict)
        checkpoint.save()
        checkpoints = os.listdir("checkpoints")
        self.assertIn("model_999.pkl", checkpoints)
        os.remove("checkpoints/model_999.pkl")


