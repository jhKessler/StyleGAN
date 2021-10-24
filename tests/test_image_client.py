import configure_pythonpath

from clients import ImageClient
import unittest
import os
import torch


class TestImageClient(unittest.TestCase):


    def test_make_image_noise(self):
        device = torch.device("cuda")
        batch_size, style_dim = 32, 256
        noise = ImageClient.make_image_noise(batch_size, style_dim, device)

        self.assertEquals(noise.shape, (5, batch_size, 2, style_dim))
        self.assertTrue(noise.is_cuda)


    def test_save_progress_images(self):
        ImageClient.save_progress_images(images = torch.randn(64, 3, 64, 64), model_id = 1, n_iterations = -1)
        folder_files = os.listdir("intermediate_images/model_1")
        self.assertIn("iteration_-1.png", folder_files)
        os.remove("intermediate_images/model_1/iteration_-1.png")

        