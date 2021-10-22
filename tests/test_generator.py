import configure_pythonpath
from models import generator
from models.generator import GenBlock
from models.generator.modules import *
from models import Generator
from clients import ImageClient

import unittest
import torch

class TestGenerator(unittest.TestCase):


    def test_base_constant(self):
        n_test_channels, test_batch_size, n_base_size = 64, 32, 4
        base_constant = BaseConstant(n_channels = n_test_channels, size = n_base_size)
        inp = torch.randn(test_batch_size, n_base_size, n_base_size)
        self.assertEquals(base_constant(inp).shape, (test_batch_size, n_test_channels, n_base_size, n_base_size))

    
    def test_mapping_network(self):
        mapping_network = MappingNetwork(n_layer = 8, layer_perceptrons = 256, bias = False)
        self.assertEquals(mapping_network(torch.randn(32, 2, 256)).shape, (32, 2, 256))

    
    def test_noise_injection(self):
        noise_injection = NoiseInjection(n_channels=256)
        output = noise_injection(torch.randn(32, 256, 4, 4), torch.randn(32, 1, 4, 4))
        self.assertEquals(output.shape, (32, 256, 4, 4))

    
    def test_adaptive_instance_normalization(self):
        ada_inst_norm = AdaptiveInstanceNormalization(128, 256)
        style = torch.randn(64, 256)
        img = torch.randn(64, 128, 4, 4)
        outp = ada_inst_norm(img, style)
        self.assertEquals(outp.shape, (64, 128, 4, 4))


    def test_gen_block(self):
        device = torch.device("cuda")
        gen_block = GenBlock(in_channels=256, out_channels=128, bias=False, initial=True).to(device)
        inp = torch.randn(32, 256, 4, 4).to(device)
        style = torch.randn(32, 2, 256).to(device)
        outp = gen_block(inp, style, device)
        self.assertEquals(outp.shape, (32, 128, 4, 4))

        gen_block = GenBlock(in_channels=256, out_channels=128, bias=False, initial=False).to(device)
        inp = torch.randn(32, 256, 4, 4).to(device)
        style = torch.randn(32, 2, 256).to(device)
        outp = gen_block(inp, style, device)
        self.assertEquals(outp.shape, (32, 128, 8, 8))


    def test_generator(self):
        device = torch.device("cuda")
        generator = Generator(256, bias = False).to(device)
        noise = ImageClient.make_image_noise(32, 256, device)
        self.assertEquals(generator(noise, device).shape, (32, 3, 64, 64))