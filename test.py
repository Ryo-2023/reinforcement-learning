import pomdp_py
import torch
import itertools
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from PBVI.util_funcs import Util_Funcs

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

def generate_belief_points(step=0.5):
    """0.2刻みで信念点の組を生成"""
    values = torch.arange(0, 1 + step, step).round(decimals=2)
    belief_combinations = torch.cartesian_prod(*[values] * 12)
    
    # 合計が1になる組み合わせのみを選択
    valid_belief_points = belief_combinations[torch.abs(belief_combinations.sum(dim=1) - 1.0) < 1e-6]
    
    if valid_belief_points.size(0) == 0:
        raise ValueError("No valid belief points generated. Check the step size and num_states.")
    
    belief_points_tensor = valid_belief_points.T.clone().detach()
    #print("belief_points:", belief_points_tensor)
    
    return belief_points_tensor

def generate_belief_points2(self,step=1/2):
    """0.2刻みで信念点の組を生成"""
    values = [round(i * step,2) for i in range(int(1/step)+1)]  
    #belief_points = torch.tensor([list(comb) for comb in itertools.product(values,repeat=num_states) if round(sum(comb),10)==1.0]).T
    
    belief_points = []
    belief_combinations = itertools.product(values, repeat=12)
    total_combinations = len(values)**12
    print("start generating belief points")
    with tqdm(total=total_combinations, desc="Generating belief points") as p_bar:
        for belief in belief_combinations:
            if abs(sum(belief) - 1.0) < 1e-6:  # 合計が1になる組み合わせのみを選択
                belief_points.append(list(belief))
            p_bar.update(1)
    
    if len(belief_points) == 0:
        raise ValueError("No valid belief points generated. Check the step size and num_states.")
    
    belief_points_tensor = torch.tensor(belief_points).T.clone().detach()
    #print("belief_points:", belief_points_tensor)
    
    return belief_points_tensor
    
state = torch.tensor([0,1,0,1,1])
state_list= torch.tensor([[0., 0., 1., 0., 0.],
                        [0., 0., 1., 0., 1.],
                        [0., 0., 1., 1., 0.],
                        [0., 0., 1., 1., 1.],
                        [0., 1., 0., 0., 0.],
                        [0., 1., 0., 0., 1.],
                        [0., 1., 0., 1., 0.],
                        [0., 1., 0., 1., 1.],
                        [1., 0., 0., 0., 0.],
                        [1., 0., 0., 0., 1.],
                        [1., 0., 0., 1., 0.],
                        [1., 0., 0., 1., 1.]])
attentions = torch.tensor([[0,1,0],[1,0,0],[0,0,1]])

action = torch.tensor([1,0,0])

obs = torch.tensor([0,1,0])
observations = torch.tensor([[1,0,0],[0,1,0],[0,0,1]])

trans_prob_others = torch.tensor([[0.7, 0.1, 0.1, 0.1],  
                                [0.1, 0.7, 0.1, 0.1],
                                [0.1, 0.1, 0.7, 0.1],
                                [0.1, 0.1, 0.1, 0.7]])

sigma = 0.7

print("start generating belief points 1")
start_time = time.time()
belief_points = generate_belief_points(1/3)
print("elapsed time 1:", time.time() - start_time)

print("start generating belief points 2")
start_time = time.time()
belief_points = generate_belief_points2(1/3)
print("elapsed time 2:", time.time() - start_time)

