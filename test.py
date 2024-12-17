import torch

a = torch.tensor([0,1,0,0])
b = a.repeat(3,4,1)
print(b)