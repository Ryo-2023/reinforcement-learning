import pomdp_py
import torch
import itertools
import pickle

with open("E:/sotsuron/venv_sotsuron/src/data/data_belief.pkl", "rb") as f:
    data = pickle.load(f)
    
print(data)