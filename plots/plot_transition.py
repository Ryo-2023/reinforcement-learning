import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import os


def plot_heatmap(data_path, save_path, title):
    # データの準備
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    # ヒートマップの作成
    plt.figure(figsize=(12, 10))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="Blues", cbar=True, 
                xticklabels=["Attention 1", "Attention 2", "Attention 3"], 
                yticklabels=np.arange(1, len(data)+1), annot_kws={"size": 10})
    plt.xlabel("Attention Layers", fontsize=14)
    plt.ylabel("Time Steps", fontsize=14)
    plt.title(f"{title} Transition (reward: hybrid)", fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # ファイルに保存
    plt.savefig(save_path)
    plt.show()
    
def plot_line(data_path, save_path, title):
    # データの準備
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    data = np.array(data)
    
    # データが2次元であることを確認
    if len(data.shape) != 2:
        raise ValueError(f"Data must be 2-dimensional. Got shape {data.shape}.")

    # 折れ線グラフの作成
    plt.figure(figsize=(12, 10))
    for i in range(data.shape[1]):
        plt.plot(data[:, i], label=f"Attention {i+1}")
    plt.xlabel("Time Steps", fontsize=14)
    plt.ylabel("Values", fontsize=14)
    plt.title(f"{title} Transition (reward: hybrid)", fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True)
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
    
    # plot
    plot_heatmap(state_data_path,state_save_path,"State")
    plot_line(action_data_path,action_save_path,"Action")
    plot_line(obs_data_path,obs_save_path,"Observation")
    
if __name__ == "__main__":
    main()

