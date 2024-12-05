import pomdp_py
import random
import itertools
import ast
import inspect as isp
import scipy
import numpy as np
import matplotlib.pyplot as plt
import torch

list = [[[1,2],[3,4]],[[5,6],[7,8]]]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tensor = torch.tensor(list, device=device)

print(tensor)