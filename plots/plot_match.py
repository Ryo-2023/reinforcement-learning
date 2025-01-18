import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os
import sys

# PBVIのモジュールパスを追加
sys.path.append("E:/sotsuron/venv_sotsuron/src/PBVI")

from util import util

# ヒートマップ重ねる
def plot_double_heatmap(data1, data2, save_path, title):
    # データが2次元であることを確認
    if len(data1.shape) != 2 or len(data2.shape) != 2:
        raise ValueError(f"Data must be 2-dimensional. Got shapes {data1.shape} and {data2.shape}.")

    # 一致率の計算
    match_count = 0
    nstep = data1.shape[0]
    for i in range(nstep):
        if np.array_equal(data1[i] , data2[i]):
            match_count += 1
    overall_match_rate = match_count / nstep
    
    # ヒートマップの作成
    plt.figure(figsize=(12, 10))
    
    # ヒートマップ1 (state)
    sns.heatmap(data1, cmap="Blues", alpha=0.7, cbar=False, annot=False)
    
    # ヒートマップ2 (action)
    sns.heatmap(data2, cmap="Reds", alpha=0.7, cbar=False, annot=False)
    
    # 重なっている部分を強調 (state と action の両方が 1 の部分)
    overlap = (data1 == 1) & (data2 == 1)
    sns.heatmap(overlap, cmap="Greens", alpha=0.5, cbar=False, annot=False)
    
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("States/Actions", fontsize=14)
    plt.title(f"{title} Transition\nOverall Match Rate: {overall_match_rate:.2%}", fontsize=16, fontweight='bold')

    plt.tight_layout()
    
    # ファイルに保存
    plt.savefig(save_path)
    plt.show()
    
    
def main():
    save_dir_path = "E:/sotsuron/venv_sotsuron/src/PBVI/PBVI_data/plot"
    
    # データのパス
    data_dir_path = "E:/sotsuron/venv_sotsuron/src/PBVI/PBVI_data/logs"
    state_data_path = os.path.join(data_dir_path,"data_state.pkl")
    action_data_path = os.path.join(data_dir_path,"data_action.pkl")
    obs_data_path = os.path.join(data_dir_path,"data_obs.pkl")
    
    # 保存先のパス
    os.makedirs(save_dir_path, exist_ok=True)
    state_save_path = os.path.join(save_dir_path,"plot_state.png")
    action_save_path = os.path.join(save_dir_path,"plot_action.png")
    obs_save_path = os.path.join(save_dir_path,"plot_obs.png")
    
    # plot double heatmap
    state_data = util.open_data(state_data_path)
    state_attentions = [state[:3] for state in state_data]
    
    action_data = util.open_data(action_data_path)
    
    plot_double_heatmap(np.array(state_attentions),np.array(action_data),os.path.join(save_dir_path,"plot_double_heatmap.png"),"State/Action")
    
if __name__ == "__main__":
    main()