import configure_pythonpath

from models import Discriminator
from models import discriminator
from models.discriminator import DiscBlock, disc_block
from models.discriminator.modules import *

import unittest
import torch

class TestDiscriminator(unittest.TestCase):


    def test_from_rgb(self):
        test_img = torch.randn(64, 3, 64, 64)
        from_rgb = FromRGB(out_channels=256)
        outp = from_rgb(test_img)
        self.assertEquals(outp.shape, (64, 256, 64, 64))


    def test_decision_block(self):
        decision_block = DecisionBlock(256)
        test_img = torch.randn(64, 256, 4, 4)
        outp = decision_block(test_img)
        self.assertEquals(outp.shape, tuple([64]))


    def test_disc_block(self):
        disc_block = DiscBlock(128, 256, bias = False)
        test_img = torch.randn(64, 128, 32, 32)
        outp = disc_block(test_img)
        self.assertEquals(outp.shape, (64, 256, 16, 16))


    def test_discriminator(self):
        discriminator = Discriminator()
        test_img = torch.randn(64, 3, 64, 64)
        outp = discriminator(test_img)
        self.assertEquals(outp.shape, tuple([64]))