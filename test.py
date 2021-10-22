from clients import TrainingClient, ImageClient
from models import Generator
import torch

img_batch = torch.randn(32, 3, 32, 32)
merged_images = TrainingClient.merge_images(img_batch, 0.5)
