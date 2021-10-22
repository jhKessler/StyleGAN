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
    def create_dataloader(batch_size: int, img_size: int, upscale: bool) -> DataLoader:
        """
        Creates dataloader that loads the images
        """
        print("Creating Dataloader...")
        data_path = "data"

        if upscale:
            img_size = img_size * 2

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
        image_batch = F.interpolate(image_batch, scale_factor = 2)
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

    
