import torch
from checkpoint_utils import get_model_dict

def train_model(Args):
    """
    training for model
    """
    
    torch.manual_seed(Args.SEED)
    Args.DEVICE = torch.cuda.device(Args.DEVICE)

    model_dict = get_model_dict(Args)
