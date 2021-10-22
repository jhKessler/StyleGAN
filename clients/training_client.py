import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as dset
import subprocess


class TrainingClient(object):
    """
    Class providing various functionality for training
    """

    @staticmethod
    def create_dataloader(batch_size: int, img_size: int) -> DataLoader:
        """
        Creates dataloader that loads the images
        """
        print("Creating Dataloader...")
        data_path = "data"

        data = dset.ImageFolder(
            root = data_path,
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                transforms.RandomHorizontalFlip(p = 0.5)
            ])
        )
        loader = DataLoader(data, batch_size, shuffle = True, num_workers = 3)
        return loader


    @staticmethod
    def merge_images(image_batch: torch.tensor, alpha: float) -> torch.tensor:
        """
        Merges images from 2 Resolutions so the transition between 2 resolutions in training is smoother
        """
        if alpha < 1:
            downscaled = F.avg_pool2d(image_batch, kernel_size = 2)
            downscaled = F.interpolate(image_batch, scale_factor = 1)
            image_batch = (1-alpha) * downscaled + image_batch * alpha
        image_batch = F.interpolate(image_batch, size = (64, 64))
        return image_batch

    
    @staticmethod
    def calculate_FID_score():
        real_data = "fid_format_data/real"
        fake_data = "fid_format_data/fake"
        print("Calculating FID Score...")
        outp = subprocess.run(["python", "-m", "pytorch_fid", real_data, fake_data, "--batch-size", "128", "--device", "cuda:0"], capture_output=True, text=True).stdout
        outp = outp.split()[1].replace("\n", "")
        print(f"FID Score: {outp}")
        return float(outp)


    @staticmethod
    def gradient_penalty(discriminator, real_images, fake_images, device):
        """gradient pentalty for gan loss fn"""
        bs, channels, height, width = real_images.shape
        eps = torch.rand(bs, 1, 1, 1).to(device).repeat(1, channels, height, width)

        # merge fake and real images
        merged_images = real_images * eps + fake_images * (1 - eps)
        merged_predict = discriminator(merged_images)

        gradient_penalty = torch.autograd.grad(
            inputs=merged_images,
            outputs=merged_predict,
            grad_outputs=torch.ones_like(merged_predict),
            create_graph=True,
            retain_graph=True
        )[0]
        gradient_penalty = gradient_penalty.view(bs, -1)
        gradient_penalty = gradient_penalty.norm(2, dim=1)
        gradient_penalty = torch.mean((gradient_penalty - 1) ** 2)
        return gradient_penalty
    
