import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle

# データの準備
with open("E:/sotsuron/venv_sotsuron/src/data/comfort/data_action.pkl", "rb") as f:
    data = pickle.load(f)

# ヒートマップの作成
plt.figure(figsize=(10, 8))
sns.heatmap(data, annot=True, cmap="YlGnBu", cbar=True, xticklabels=["Attention 1", "Attention 2", "Attention 3"], yticklabels=np.arange(1, len(data)+1))
plt.xlabel("Attention Layers")
plt.ylabel("Time Steps")
plt.title("Attention Layer Transitions Over Time")

# ファイルに保存
plt.savefig("E:/sotsuron/venv_sotsuron/src/data/comfort/action_transition.png")  # reward = mse
plt.show()

