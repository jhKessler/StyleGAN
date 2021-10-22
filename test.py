from models import Generator
import torch

g = torch.randn(1, 2).to(torch.device("cuda"))
print(g.is_cuda)