import pomdp_py
import torch
import itertools
import pickle
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from PBVI.Util_Funcs import util_funcs

torch.set_default_dtype(torch.float32)
torch.set_default_tensor_type = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)

class Agent:
    def __init__(self, num_states, num_actions, num_observations):
        self.num_states = num_states
        self.num_actions = num_actions
        self.num_observations = num_observations
        self.trans_prob = torch.rand(num_states, num_states)  # ダミーの遷移確率
        self.obs_prob = torch.rand(num_states, num_actions, num_observations)  # ダミーの観測確率
        self.belief = (torch.ones(num_states) / num_states).repeat(self.num_actions,self.num_observations,1)  # 初期信念状態

    def update_belief(self, belief):
        # 入力には[a_{t-1},o_{t-1}]のときの信念b(s):[s]が入る
        # 正規化項 reg
        sum_s1 = torch.einsum("sj,aos->aoj", self.trans_prob, belief)  # [s,s'] * [a,o,s] -> [a,o,s']
        sum_s2 = torch.einsum("sj,jao->sao", self.trans_prob, self.obs_prob)  # [s,s'] * [s',a,o] -> [s,a,o]

        # 正規化項
        reg = torch.einsum("aos,sao->ao", belief, sum_s2)  # [a,o,s] * [s,a,o] -> [a,o]

        # 信念の更新
        belief_not_reg = torch.einsum("jao,aoj->aoj", self.obs_prob, sum_s1)  # [s,a,o] * [a,o,s] -> [a,o,s]
        update_belief = belief_not_reg / reg.unsqueeze(-1).expand(-1, -1, self.num_states)  # [a,o,s] / ([a,o] -> [a,o,s])

        """
        # デバッグ用に中間結果を表示
        print("sum_s1:", sum_s1)
        print("sum_s2:", sum_s2)
        print("reg:", reg)
        print("belief_not_reg:", belief_not_reg)
        print("update_belief:", update_belief)
        """
        
        self.belief = update_belief

        return update_belief

# デバッグ用のデータを準備
num_states = 5
num_actions = 3
num_observations = 2

agent = Agent(num_states, num_actions, num_observations)

# ダミーの信念状態
belief = torch.rand(num_actions, num_observations, num_states)

# update_belief 関数の呼び出し
for _ in range(10):
    updated_belief = agent.update_belief(agent.belief)

    # 結果の表示
    print("Updated Belief State:")
    print(updated_belief)
