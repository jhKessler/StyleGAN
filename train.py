import torch
from clients import CheckpointClient, ImageClient

def train_model(Args):
    """
    training for model
    """
    
    torch.manual_seed(Args.SEED)
    Args.DEVICE = torch.device(Args.DEVICE)

    checkpoint = CheckpointClient.get_checkpoint(Args)
    model_dict = checkpoint.get_dict()
    generator = model_dict["generator"]
    noise = ImageClient.make_image_noise(1, Args.BATCH_SIZE, Args.NOISE_DIM, Args.DEVICE)



    
