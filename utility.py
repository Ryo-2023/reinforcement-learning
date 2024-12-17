import pomdp_py
import random
import itertools
import ast
import inspect as isp
import scipy
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

class Model():
    def __init__(self, sigma, num_people):
        self.sigma = sigma
        self.n = num_people
        
        # 状態、行動、観測空間の生成
        self.attentions = self.generate_weight_list(self.n, 1)
        self.comforts = torch.tensor([0, 1])
        self.follows = torch.tensor([0, 1])
        
        # すべての組み合わせを生成
        """
        state[0] ~ state[n-1] : attention
        state[n]   : comfort
        state[n+1] : follow
        states: tensor([[0., 0., 1., 0., 0.],
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
                        [1., 0., 0., 1., 1.]]) shape:(12, 5)
        """
        self.states = self.generate_state_space()
        self.actions = self.generate_weight_list(self.n,2)
        self.observations = self.generate_weight_list(self.n,1)
        
        self.num_attentions = self.attentions.shape[0]
        self.num_comforts = self.comforts.shape[0]
        self.num_follows = self.follows.shape[0]
        
        self.num_states = self.states.shape[0]
        self.num_actions = self.actions.shape[0]
        self.num_observations = self.observations.shape[0]
        
        # 状態遷移確率の初期化
        self.trans_prob_attention = torch.eye(self.num_attentions) # dim1 : current attention (縦軸), dim2 : next attention (横軸)
        for i  in range(self.num_attentions):
            self.trans_prob_attention[i] = self.pdf_list(self.attentions[i],self.sigma,self.attentions)  # 状態遷移確率は正規分布でモデル化
        
        # dim1 : current[comfort, follow] (縦軸), dim2 : next[comfort, follow] (横軸) 4*4の行列 [00,01,10,11]
        self.trans_prob_others = torch.tensor([[0.7, 0.1, 0.1, 0.1],  
                                               [0.1, 0.7, 0.1, 0.1],
                                               [0.1, 0.1, 0.7, 0.1],
                                               [0.1, 0.1, 0.1, 0.7]])
                    
        # attentionとothersの遷移確率を結合
        self.trans_prob = torch.zeros((self.num_states,self.num_states))
        self.trans_prob = torch.einsum("ij,kl->ikjl",self.trans_prob_attention,self.trans_prob_others).reshape(self.num_states,self.num_states)
        """
        # einsumを使わない場合
        for i in range(self.num_attentions): 
            for j in range(self.num_attentions): 
                start_i = i*self.num_comfort*self.num_follows
                end_i = (i+1)*self.num_comfort*self.num_follows
                start_j = j*self.num_comfort*self.num_follows
                end_j = (j+1)*self.num_comfort*self.num_follows
                self.trans_prob[start_i:end_i,start_j:end_j] = self.trans_prob_attention[i,j] * self.trans_prob_others
        """
                
        # 観測確率の初期化
        self.obs_prob = self.init_obs_prob()
        
        # 信念状態の初期化
        self.belief = torch.ones(self.num_states) / self.num_states
        
    # 報酬関数の定義
    def reward(self,state):
        return self.state[self.n+1] # 報酬 : comfort
    
    def generate_weight_list(self,n_people,step):
        values = [round(i * 1/step, 2) for i in range(step+1)]  # [0.0, 0.5, 1.0] のリスト
        return torch.tensor([list(comb) for comb in itertools.product(values, repeat=n_people) if round(sum(comb),10)==1.0])
    
    def generate_state_space(self):
        # すべての組み合わせを生成
        state_combinations = list(itertools.product(self.attentions, self.comforts, self.follows))
        
        # テンソルに変換
        state_space = torch.tensor([list(state[0]) + [state[1]] + [state[2]] for state in state_combinations])
        
        return state_space
    
    def pdf_list(self, mean, sigma, weight_list):  # 確率密度関数の生成
        probabilities = []
        for weights in weight_list:
            diff = weights - mean
            prob = (torch.exp(-0.5 * ((diff / sigma) ** 2))/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))).prod().item()
            probabilities.append(prob)
        normalized_probabilities = torch.tensor([prob / sum(probabilities) for prob in probabilities])
        return normalized_probabilities
    
    def init_obs_prob(self):
        tensor_shape = (self.num_states, self.num_actions, self.num_observations)
        obs_prob = torch.zeros(tensor_shape)
        
        # 元の順番に戻すためのインデックスを保持, 復元に必要
        follow_0_index = (self.states[:, -1] == 0).nonzero(as_tuple=True)[0]
        follow_1_index = (self.states[:, -1] == 1).nonzero(as_tuple=True)[0]
        
        print(follow_0_index)
        # follow = 1, システムに従う場合
        for i in follow_1_index:
            for j in range(self.num_actions):
                prob = 0.1 * self.pdf_list(self.states[i, :self.n], self.sigma, self.observations) + 0.9 * self.pdf_list(self.actions[j], self.sigma, self.observations)
                obs_prob[i,j,:] = prob
        # follow = 0, システムに従わない場合
        for i in follow_0_index:
            for j in range(self.num_actions):
                prob = 0.9 * self.pdf_list(self.states[i, :self.n], self.sigma, self.observations) + 0.1 * self.pdf_list(self.actions[j], self.sigma, self.observations)
                obs_prob[i,j,:] = prob
        """
        for i in range(self.num_states):
            for j in range(self.num_actions):
                prob = 0.1 * self.pdf_list(self.states[i, :self.n], self.sigma, self.observations) + 0.9 * self.pdf_list(self.actions[j], self.sigma, self.observations)
                self.obs_prob_follow[i, j, :] = prob
                    
        # follow = 0, システムに従わない場合
        for i in range(self.num_states):
            for j in range(self.num_actions):
                prob = 0.9 * self.pdf_list(self.states[i, :self.n], self.sigma, self.observations) + 0.1 * self.pdf_list(self.actions[j], self.sigma, self.observations)
                self.obs_prob_follow[i, j, :] = prob
        """
        return obs_prob
        
# 使用例
model = Model(sigma=0.3, num_people=3)
#print("attentions:", model.attentions)
#print("states:", model.states)
#print("actions:", model.actions)
#print("num_actions:", model.num_actions)
#print("observations:", model.observations)
sum_s = torch.einsum("i,ij->j",model.belief,model.trans_prob)
sum_s2 = torch.einsum("i,ijk->ijk",sum_s,model.obs_prob)
print(sum_s.shape)
print(sum_s2)
