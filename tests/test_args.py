import unittest
import os
import configure_pythonpath
from args import Args
import torch

class TestArgs(unittest.TestCase):


    def test_fade_size(self):
        # check fade size
        self.assertIsInstance(Args.FADE_SIZE, int)
        self.assertGreater(Args.FADE_SIZE, 0)


    def test_batch_size(self):
        # check batch size
        self.assertIsInstance(Args.BATCH_SIZE, int)


    def test_noise_dim(self):
        # check noise dimension
        self.assertIsInstance(Args.NOISE_DIM, int)
        self.assertGreater(Args.NOISE_DIM, 0)


    def test_resolution(self):
        # check resolution
        self.assertIsInstance(Args.RESOLUTION, int)
        self.assertGreater(Args.RESOLUTION, 0)


    def test_num_progress_img(self):
        # check resolution
        self.assertIsInstance(Args.NUM_PROGRESS_IMGS, int)
        self.assertGreater(Args.NUM_PROGRESS_IMGS, 0)


    def test_learning_rate(self):
        # check learning rate
        self.assertIsInstance(Args.LR, float)
        self.assertGreater(Args.LR, 0)


    def test_device(self):
        # check device
        self.assertIn(Args.DEVICE, ["cpu", "cuda"])
        if Args.DEVICE == "cuda":
            self.assertTrue(torch.cuda.is_available())

    def test_data_path(self):
        # check data path
        self.assertTrue(os.path.isdir(Args.DATA_PATH))


    def test_betas(self):
        # check betas
        self.assertIsInstance(Args.BETAS, list)
        self.assertEquals(2, len(Args.BETAS))
        for val in Args.BETAS:
            self.assertIsInstance(val, float)
        
