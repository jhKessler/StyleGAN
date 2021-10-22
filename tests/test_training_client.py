import configure_pythonpath
from clients import TrainingClient
import unittest
import torch


class TestTrainingClient(unittest.TestCase):


    def test_create_dataloader(self):
        batch_size, img_size = 32, 32
        dataloader = TrainingClient.create_dataloader(batch_size, img_size)
        batch = next(iter(dataloader))[0]
        self.assertEquals(batch.shape, (32, 3, img_size, img_size))

        img_size = 64
        dataloader = TrainingClient.create_dataloader(batch_size, img_size)
        batch = next(iter(dataloader))[0]
        self.assertEquals(batch.shape, (32, 3, img_size, img_size))

        img_size = 8
        dataloader = TrainingClient.create_dataloader(batch_size, img_size)
        batch = next(iter(dataloader))[0]
        self.assertEquals(batch.shape, (32, 3, img_size, img_size))
        

    def test_merge_images(self):
        img_batch = torch.randn(32, 3, 32, 32)
        merged_images = TrainingClient.merge_images(img_batch, 0.5)
        self.assertEquals(merged_images.shape, (32, 3, 64, 64))
