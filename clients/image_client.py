import os
import torch
import numpy as np
import torchvision.utils as vutils
import matplotlib.pyplot as plt


class ImageClient(object):
    """
    Client offering multiple image Operations e.g. generating and saving images with generator model 
    """

    @staticmethod
    def make_image_noise(batch_size: int, noise_dim: int, device: torch.device) -> torch.tensor:
        gen_block_count = 5
        conv_block_count = 2
        return torch.randn(gen_block_count, batch_size, conv_block_count, noise_dim).to(device)


    @staticmethod
    def save_progress_images(images: torch.tensor, model_id: int, n_iterations: int) -> None:
        """
        Save progress images for making gif after training 
        """
        # decouple images
        images = images.detach().cpu()

        # create image folder if not already existing
        folder_path = os.path.join("intermediate_images", f"model_{model_id}")
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        # make image grid
        image_grid = np.transpose(vutils.make_grid(images, padding = 2, normalize = True), (1, 2, 0))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Progress")
        plt.imshow(image_grid)
        plt.savefig(os.path.join(folder_path, f"iteration_{n_iterations}.png"))
        plt.close()
        
    