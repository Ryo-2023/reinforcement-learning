import pickle
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import os
import sys

# PBVIのモジュールパスを追加
sys.path.append("E:/sotsuron/venv_sotsuron/src/PBVI")

from util import util

class plot_belief:
    def __init__(self,data_path,save_path):
        # データのパス
        self.data_dir_path = data_path
        
        # 保存先のパス
        self.save_dir_path = save_path
    
    def load_data(self,filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    def reshape_data(self, data):
        states_list = list(range(len(data[0])))  # 状態のリストを作成
        values_list = []
        steps = []
        nsteps = len(data)
        for i in range(nsteps):
            value_data = data[i]
            values_list.append(value_data)
            steps.append((states_list, value_data))
        return states_list, values_list, steps, nsteps

    def update(self, frame, states, values, state, action, obs):
        plt.cla()
        plt.barh(states, values[frame])  # 横棒グラフ
        plt.xlim(0, 1)
        plt.xlabel("belief")
        plt.ylabel("state")
        plt.title(f"Belief,   frame: {frame+1}")
        plt.grid()
        
        # 各ステップ時の state, action, obs を右上に表示
        state_text = f"State: {state[frame]}"
        action_text = f"Action: {action[frame]}"
        obs_text = f"Obs: {obs[frame]}"
        plt.text(1, 0.95, state_text, transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(1, 0.90, action_text, transform=plt.gca().transAxes, verticalalignment='top')
        plt.text(1, 0.85, obs_text, transform=plt.gca().transAxes, verticalalignment='top')
        
    def plot_belief(self):
        state = self.load_data(os.path.join(self.data_dir_path,"data_state.pkl"))
        belief = self.load_data(os.path.join(self.data_dir_path,"data_belief.pkl"))
        action = self.load_data(os.path.join(self.data_dir_path,"data_action.pkl"))
        obs = self.load_data(os.path.join(self.data_dir_path,"data_obs.pkl"))
        
        states, values, steps, nsteps = self.reshape_data(belief)
        
        fig = plt.figure(figsize=(15, 6))
        ani = FuncAnimation(fig, self.update, fargs=(states, values,state,action,obs), frames=nsteps, interval = 5000, repeat=False)
        
        # gif で保存
        writer = PillowWriter(fps=2)
        ani.save(os.path.join(self.save_dir_path,"plot_belief.gif"), writer=writer)
        
        plt.show()
        
def main():
    data_dir_path = "E:/sotsuron/venv_sotsuron/src/PBVI/PBVI_data/logs"
    save_dir_path = "E:/sotsuron/venv_sotsuron/src/PBVI/PBVI_data/plot"
    
    plot_belief(data_dir_path,save_dir_path).plot_belief()
    
if __name__ == "__main__":
    main()
    