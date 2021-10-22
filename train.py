import torch
from clients import CheckpointClient, ImageClient
import utils

def train_model(Args):
    """
    training for model
    """
    
    torch.manual_seed(Args.SEED)
    Args.DEVICE = torch.device(Args.DEVICE) if isinstance(Args.DEVICE, str) else Args.DEVICE
    
    checkpoint = CheckpointClient.get_checkpoint(Args)
    model_dict = checkpoint.get_dict()


    generator = model_dict["generator"]
    g_optim = model_dict["g_optimizer"]
    discriminator = model_dict["discriminator"]
    d_optim = model_dict["d_optimizer"]
    preview_noise = model_dict["preview_noise"]

    iteration = model_dict["iteration"]
    samples = model_dict["samples"]
    step = model_dict["step"]
    alpha = model_dict["alpha"]


    for step in range(Args.MAX_STEPS):
        img_size = 4 * (2 ** step)
        print("Starting training for resolution {img_size}x{img_size}...")

        for current_iteration in range(PHASE_SIZE)
    
    
    




    
