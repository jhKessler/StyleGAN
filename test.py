from clients import TrainingClient, ImageClient
from models import Generator
import torch

noise = ImageClient.make_image_noise(128, 256, torch.device("cuda"))
g = Generator(256).to(torch.device("cuda"))

ImageClient.save_fid_images(g(noise, torch.device("cuda")))
