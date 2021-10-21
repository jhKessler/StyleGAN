import torch
from clients.checkpoint_client import CheckpointClient

def train_model(Args):
    """
    training for model
    """
    
    torch.manual_seed(Args.SEED)
    Args.DEVICE = torch.cuda.device(Args.DEVICE)

    checkpoint = CheckpointClient.get_checkpoint(Args)
    model_dict = checkpoint.get_dict()

    
