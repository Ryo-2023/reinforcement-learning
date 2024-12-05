import torch
import itertools
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import pickle
import numpy as np

class State:
    def __init__(self, user, env):
        self.user = user
        self.env = env
        
    def __repr__(self):
        return "Model_State(user=%s, environment=%s)" % (self.user, self.env)
    
    def __hash__(self):
        return hash((tuple(self.user), tuple(self.env)))
    
    def __eq__(self, other):
        return self.user == other.user and self.env == other.env
    
    def get_all_states(self):
        state_list = generate_weight_list()
        all_states = []
        for env in state_list:
            for user in state_list:
                all_states.append(State(env, user))
        return all_states

class Action:
    def __init__(self, action):
        self.action = action
        
    def __repr__(self):
        return "Model_Action(action=%s)" % (self.action)
    
    def get_all_actions(self):
        action_list = generate_weight_list()
        all_actions = []
        for action in action_list:
            all_actions.append(Action(action))
        return all_actions
        
def generate_weight_list():
    values = [round(i * 0.2, 2) for i in range(6)]
    return [[x, y, z] for x, y, z in itertools.product(values, repeat=3) if round(x + y + z, 10) == 1.0]

def generate_all_states():
    weight_list = generate_weight_list()
    all_states = []
    for env in weight_list:
        for user in weight_list:
            all_states.append(State(env, user))
    return all_states

class TransitionModel:
    def __init__(self, alpha=0.5, beta=0.5, sigma=0.1):
        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.state_list = generate_weight_list()

    def probability(self, next_state, state, action, device="cpu"):
        # environment
        env_mean = self.alpha * torch.tensor(state.env) + (1 - self.alpha) * torch.tensor(action.action)
        normalized_probabilities = pdf_list(env_mean, self.sigma, self.state_list)
        next_env_probability = normalized_probabilities[self.state_list.index(next_state.env)]

        # user
        user_mean = self.beta * torch.tensor(state.user) + (1 - self.beta) * torch.tensor(action.action)
        normalized_probabilities = pdf_list(user_mean, self.sigma, self.state_list)
        next_user_probability = normalized_probabilities[self.state_list.index(next_state.user)]
        
        return [next_env_probability, next_user_probability]
            
    def sample(self, state, action, device="cpu"):
        # environment
        env_mean = self.alpha * torch.tensor(state.env) + (1 - self.alpha) * torch.tensor(action.action)
        normalized_probabilities = pdf_list(env_mean, self.sigma, self.state_list)
        next_env_sample = self.state_list[torch.multinomial(torch.tensor(normalized_probabilities), 1).item()]

        # user
        user_mean = self.beta * torch.tensor(state.user) + (1 - self.beta) * torch.tensor(action.action)
        normalized_probabilities = pdf_list(user_mean, self.sigma, self.state_list)
        next_user_sample = self.state_list[torch.multinomial(torch.tensor(normalized_probabilities), 1).item()]

        return State(next_env_sample, next_user_sample)

class RewardModel:
    def reward_func(self, state, action):
        mse = mean_squared_error(state.user, action.action)
        return -mse

class MDP:
    def __init__(self, state, action):
        self.state = state
        self.action = action
        self.transition_model = TransitionModel()
        self.reward_model = RewardModel()
        self.all_states = self.state.get_all_states()
        self.all_actions = self.action.get_all_actions()
        
    def ValueIteration(self, threshold=1e-3, discount_factor=0.9):
        V = {s: 0 for s in self.all_states}  # 価値関数の初期化
        policy = {s: None for s in self.all_states}  # 方策の初期化
        
        while True:
            delta = 0
            with tqdm(total=len(self.all_states), desc="Value Iteration") as pbar:
                with ProcessPoolExecutor(max_workers=2) as executor:  # プロセスの数を制限
                    futures = []
                    for s in self.all_states:
                        futures.append(executor.submit(self._evaluate_state, s, pickle.dumps(V), discount_factor))
                    for future in as_completed(futures):
                        s, max_value, best_action = future.result()
                        delta = max(delta, abs(V[s] - max_value))
                        V[s] = max_value
                        policy[s] = best_action
                        pbar.update(1)
            if delta < threshold:
                break
        return V, policy

    def _evaluate_state(self, s, V_serialized, discount_factor):
        V = pickle.loads(V_serialized)
        max_value = float("-inf")
        best_action = None
        for action in self.all_actions:
            expected_value = 0
            for next_state in self.all_states:
                transition_prob = self.transition_model.probability(next_state, s, action)
                reward = self.reward_model.reward_func(next_state, action)
                expected_value += transition_prob[0] * transition_prob[1] * (reward + discount_factor * V[next_state])
            if expected_value > max_value:
                max_value = expected_value
                best_action = action
        return s, max_value, best_action

def pdf_list(mean, sigma, state_list):
    return [np.exp(-np.sum((np.array(mean) - np.array(state))**2) / (2 * sigma**2)) for state in state_list]

# 使用例
state = State([0.4, 0.4, 0.2], [0.3, 0.4, 0.3])
action = Action([0.3, 0.5, 0.2])
mdp = MDP(state, action)
V, policy = mdp.ValueIteration()
print("Value Function:", V)
print("Policy:", policy)