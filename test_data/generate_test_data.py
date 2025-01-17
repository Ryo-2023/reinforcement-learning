import pickle
import os

date = [
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
    [1,0,0],
]

file_path = "E:/sotsuron/venv_sotsuron/src/data/test_data.pkl"

# pickle 形式で保存
with open(file_path, "wb") as f:
    pickle.dump(date, f)

print("data saved as pickle file in : ", file_path)
