import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# データの準備
data = [
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [0,0,1],
    [0,1,0],
    [0,1,0],
    [0,0,1],
    [0,0,1],
    [0,0,1],
    [0,0,1],
    [0,0,1],
    [0,0,1],
    [1,0,0],
    [1,0,0],
    [0,1,0],
    [0,1,0],
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [1,0,0],
    [1,0,0]
]

# ヒートマップの作成
plt.figure(figsize=(10, 8))
sns.heatmap(data, annot=True, cmap="YlGnBu", cbar=True, xticklabels=["Attention 1", "Attention 2", "Attention 3"], yticklabels=np.arange(1, len(data)+1))
plt.xlabel("Attention Layers")
plt.ylabel("Time Steps")
plt.title("Attention Layer Transitions Over Time")
plt.show()