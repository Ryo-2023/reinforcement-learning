import pickle
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def reshape_data(data):
    states_list = [s for s in data[0].keys()]
    values_list = []
    steps = []
    nsteps = len(data)
    for i in range(nsteps):
        value_data = [v for v in data[i].values()]
        values_list.append(value_data)
        steps.append((states_list, value_data))
    return states_list, values_list, steps, nsteps

def updata(frame, states, values,state, action, obs):
    plt.cla()
    plt.barh(states, values[frame])  # 横棒グラフ
    plt.xlim(0, 1)
    plt.xlabel("belief")
    plt.ylabel("state")
    plt.title(f"Belief (reward:mse),   frame: {frame+1}")
    plt.grid()
    
    # 各ステップ時の state, action, obs を右上に表示
    state_text = f"State: {state[frame]}"
    action_text = f"Action: {action[frame]}"
    obs_text = f"Obs: {obs[frame]}"
    plt.text(1, 0.95, state_text, transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(1, 0.90, action_text, transform=plt.gca().transAxes, verticalalignment='top')
    plt.text(1, 0.85, obs_text, transform=plt.gca().transAxes, verticalalignment='top')
    
state = load_data("E:/sotsuron/venv_sotsuron/src/data/mse/data_state.pkl")
belief = load_data("E:/sotsuron/venv_sotsuron/src/data/mse/data_belief.pkl")
action = load_data("E:/sotsuron/venv_sotsuron/src/data/mse/data_action.pkl")
obs = load_data("E:/sotsuron/venv_sotsuron/src/data/test_data.pkl")

states, values, steps, nsteps = reshape_data(belief)

fig = plt.figure(figsize=(15, 6))
ani = FuncAnimation(fig, updata, fargs=(states, values,state,action,obs), frames=nsteps, interval = 5000, repeat=False)

# gif で保存
writer = PillowWriter(fps=1)
ani.save("E:/sotsuron/venv_sotsuron/src/data/mse/plot_belief_mse.gif", writer=writer)

plt.show()
