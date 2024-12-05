import torch
import itertools
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

class State:
    def __init__(self,user,env):
        self.user = user
        self.env = env
        
    def __repr__(self):
        return "Model_State(user=%s, environment=%s)" % (self.user, self.env)
    
    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)
        """
        state_list = generate_weight_list()
        all_states = []
        for env in state_list:
            for user in state_list:
                all_states.append(State(env,user))
        return all_states
    
class Action:
    def __init__(self,action):
        self.action = action
        
    def __repr__(self):
        return "Model_Action(action=%s)" % (self.action)
    
    def get_all_actions(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)
        """
        action_list = generate_weight_list()
        all_actions = []
        for action in action_list:
            all_actions.append(Action(action))
        return all_actions
        
def generate_weight_list():
    values = [round(i * 0.2, 2) for i in range(6)]
    return [[x,y,z] for x,y,z in itertools.product(values, repeat=3) if round(x+y+z,10)==1.0]

def generate_all_states():
    weight_list = generate_weight_list()
    all_states = []
    for env in weight_list:
        for user in weight_list:
            all_states.append(State(env, user))
    return all_states

def pdf_list(mean, sigma, weight_list):  # 確率密度関数の生成
    probabilities = []
    for weights in weight_list:
        weights_tensor = torch.tensor(weights)
        diff = weights_tensor - mean.clone().detach()
        prob = (torch.exp(-0.5 * ((diff / sigma) ** 2))/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))).prod().item()
        probabilities.append(prob)
    normalized_probabilities = [prob / sum(probabilities) for prob in probabilities]
    return normalized_probabilities

class TransitionModel:
    def __init__(self, sigma=0.1,alpha=0.8,beta=0.8):
        self.sigma = sigma # 分散
        self.alpha = alpha # 重み
        self.beta  = beta  # 重み
        self.state_list = generate_weight_list()    # 状態のリストを生成

    def probability(self, next_state, state, action, device="cpu"):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, stays the same"""
        # environment
        env_mean = self.alpha * torch.tensor(state.env) + (1 - self.alpha) * torch.tensor(action.action)
        normalized_probabilities = pdf_list(env_mean, self.sigma, self.state_list)
        next_env_probability = normalized_probabilities[self.state_list.index(next_state.env)]
        #print("next_env_probability:", next_env_probability)
            
        # user
        user_mean = self.beta * torch.tensor(state.user) + (1 - self.beta) * torch.tensor(action.action)
        normalized_probabilities = pdf_list(user_mean, self.sigma, self.state_list)
        next_user_probability = normalized_probabilities[self.state_list.index(next_state.user)]
        #print("next_user_probability:", next_user_probability)
        
        return [next_env_probability, next_user_probability]
            
    def sample(self, state, action,device="cpu"):
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
    def reward_func(self,state,action):
        mse = mean_squared_error(state.user, action.action)
        return -mse

class MDP:
    def __init__(self,state,action):
        self.state = state
        self.action = action
        self.transition_model = TransitionModel()
        self.reward_model = RewardModel()
        self.all_states = self.state.get_all_states()
        self.all_actions = self.action.get_all_actions()
        
    def ValueIteration(self,threshold=1e-3,discount_factor=0.9):  # 価値反復法
        # デバイスの設定
        torch.set_default_dtype(torch.float32)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_default_device(device)
        
        V = {s: 0 for s in self.all_states}  # 価値関数の初期化
        policy  = {s: None for s in self.all_states}  # 方策の初期化
        
        while True:
            delta = 0
            with tqdm(total=len(self.all_states)) as pbar:
                for s in self.all_states:
                    v = V[s]
                    max_value = float("-inf")
                    best_action = None
                    with tqdm(total=len(self.all_actions), desc="Actions", leave=False) as pbar_action:
                        for action in self.all_actions:
                            expected_value = 0
                            with tqdm(total=len(self.all_states), desc="Next States", leave=False) as pbar_next_state:
                                for next_state in self.all_states:
                                    transition_prob = self.transition_model.probability(next_state, s, action)
                                    reward = self.reward_model.reward_func(next_state, action)
                                    expected_value += transition_prob[0] * transition_prob[1] * (reward + discount_factor * V[next_state])
                                    pbar_next_state.update(1)
                            if expected_value > max_value:
                                max_value = expected_value
                                best_action = action
                            pbar_action.update(1)
                    V[s] = max_value
                    policy[s] = best_action
                    delta  = max(delta, abs(v - V[s]))
                    pbar.update(1)
            if delta < threshold:
                break
        return V, policy

state = State([0.4,0.4,0.2],[0.2,0.6,0.2])
action = Action([0.2,0.6,0.2])
mdp = MDP(state,action)
V, policy = mdp.ValueIteration()
print("Value Function:", V)
print("Policy:", policy)