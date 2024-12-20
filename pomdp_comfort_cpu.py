import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import torch
import sys
import copy
import scipy
import sklearn.metrics
import itertools
import ast
from tqdm import tqdm
import statistics
import numpy as np
import psutil
from pomdp_py.utils import TreeDebugger
import torch.multiprocessing as mp
from pomdp_py.framework.planner import Planner
from pomdp_py.framework.basics import Agent, Action, State
from pomdp_py.representations.distribution.histogram import Histogram

class Model_State(pomdp_py.State):
    """
    state =
    {
        "environment":{
            "speaking_activity" : 発話活性度 * 視界の人数  ex.) [0.15,0.8,0.15] }
        "user": {
            "attention_weight" : ユーザーの注意の向き方 * 視界の人数 ex.) [0.15,0.8,0.15] }
    }
    """
    def __init__(self, state,step=1,n_people=3):
        self.step = step
        self.n = n_people
        
        if isinstance(state,list):
            state = torch.tensor(state)
            
        self.attention = state[:self.n]
        self.comfort = state[self.n].item()
        self.follow = state[self.n+1].item()
        self.state = state
    
        # 状態、行動、観測空間の生成
        self.attentions = generate_weight_list(self.step,self.n)
        self.comforts = torch.tensor([0, 1])
        self.follows = torch.tensor([0, 1])

    def __hash__(self):
        return hash(tuple(self.state))

    def __eq__(self, other):
        if isinstance(other, Model_State):
            return torch.equal(self.attention,other.attention) and self.comfort == other.comfort and self.follow == other.follow
        return False

    def __str__(self):  # クラスの中身を表示
        return "attention:" + str(self.attention) + " comfort:" + str(self.comfort) + " follow:" + str(self.follow)

    def __repr__(self):
        return "Model_State(attention=%s, comfort=%s, follow = %s)" % (self.attention, self.comfort, self.follow)
    
    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the state space (e.g. value iteration)
        """
        return generate_state_space(self.attentions, self.comforts, self.follows)
    
        
class Model_Action(pomdp_py.Action):
    """
    Action = {
        "weight" : どのフィルタを重視するか 重みづけ
    }
    """
    def __init__(self, enhance_weight,step=1):
        if isinstance(enhance_weight,list):
            enhance_weight = torch.tensor(enhance_weight)
            
        self.enhance_weight = enhance_weight
        self.step = step

    def __hash__(self):
        return hash(self.enhance_weight)

    def __eq__(self, other):
        if isinstance(other, Model_Action):
            return self.enhance_weight == other.enhance_weight
        return False

    def __str__(self):
        return self.enhance_weight

    def __repr__(self):
        return "Model_Action(%s)" % self.enhance_weight
    
    
class Model_Observation(pomdp_py.Observation):
    """
    Observation =
    {
        "sight_direction" : 視線の向き (one-hot vector)
    }
    """
    def __init__(self, sight_direction,step=1,n_people=3):
        if isinstance(sight_direction,list):
            sight_direction = torch.tensor(sight_direction)
            
        self.sight_direction = sight_direction
        self.step = step
        self.n = n_people

    def __hash__(self):
        return hash(tuple(self.sight_direction))

    def __eq__(self, other):
        if isinstance(other, Model_Observation):
            return self.sight_direction == other.sight_direction
        return False

    def __str__(self):
        return "sight_direction:" + str(self.sight_direction)

    def __repr__(self):
        return "Model_Observation(%s)" % self.sight_direction
    
    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space
        (e.g. value iteration)"""
        all_observations = []
        all_obs = generate_weight_list(self.step,self.n)
        for obs in all_obs:
            all_observations.append(Model_Observation(obs))
        return all_observations
    
def generate_weight_list(step=1,n_people=3):
    values = [round(i * 1/step, 2) for i in range(step+1)]  # {0,1}
    return torch.tensor([list(comb) for comb in itertools.product(values, repeat=n_people) if round(sum(comb),10)==1.0])

def generate_state_space(attentions, comforts, follows):
    # すべての組み合わせを生成
    state_combinations = list(itertools.product(attentions, comforts,follows))
    
    # テンソルに変換
    state_space = torch.tensor([list(state[0]) + [state[1]] + [state[2]] for state in state_combinations])
    
    return state_space

def pdf_list(mean, sigma, weight_list):  # 確率密度関数の生成
    probabilities = []
    # forループのため、tensor型ならリストに変換
    if isinstance(weight_list,torch.Tensor):
        weight_list = weight_list.tolist()
    for weights in weight_list:
        diff = torch.tensor(weights) - mean
        prob = (torch.exp(-0.5 * ((diff / sigma) ** 2))/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))).prod().item()
        probabilities.append(prob)
    normalized_probabilities = torch.tensor([prob / sum(probabilities) for prob in probabilities])
    return normalized_probabilities
    
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self,step = 1,n_people=3, sigma=0.1):
        self.sigma = sigma # 分散
        self.step = step
        self.n = n_people
        self.obs_list = generate_weight_list(self.step,self.n)    # 観測のリストを生成

    def probability(self, obs, next_state, action):  # follow = 1
        if next_state.follow == 1:
            normalized_prob = pdf_list(action.enhance_weight, self.sigma, self.obs_list)
            sight_prob_follow = normalized_prob[self.obs_list.index(obs.sight_direction)]  # 観測確率
            return sight_prob_follow
        elif next_state.follow == 0:
            normalized_prob = pdf_list(next_state.attention, self.sigma, self.obs_list)
            sight_prob_not_follow = normalized_prob[self.obs_list.index(obs.sight_direction)]  # 観測確率
            return sight_prob_not_follow

    def sample(self, next_state, action):  # 観測のサンプリング
        if next_state.follow == 1:
            normalized_prob = pdf_list(action.enhance_weight, self.sigma, self.obs_list)
            sight_sample_follow = self.obs_list[torch.multinomial(torch.tensor(normalized_prob), 1).item()]
            return Model_Observation(sight_sample_follow)
        elif next_state.follow == 0:
            normalized_prob = pdf_list(next_state.attention, self.sigma, self.obs_list)
            sight_sample_not_follow = self.obs_list[torch.multinomial(torch.tensor(normalized_prob), 1).item()]    # torch.multinomial:与えられた確率分布に基づきサンプリング
            return Model_Observation(sight_sample_not_follow)
        
    def get_all_observations(self):
        OBS = [Model_Observation(o) for o in self.obs_list]
        return OBS
    
class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self, step=1,n_people=3, sigma=0.1):
        self.sigma = sigma # 分散
        self.step = step
        self.n = n_people
        
        # 状態空間の生成
        self.attention_list = generate_weight_list(self.step,self.n)
        self.comforts = torch.tensor([0,1])
        self.follows  = torch.tensor([0,1])
        self.others_list = torch.tensor([[0,0],[0,1],[1,0],[1,1]])
        self.others_trans_prob_table = torch.tensor([[0.8,0.2,0,0],
                                                    [0.15,0.7,0.05,0.1],
                                                    [0.05,0.05,0.7,0.2],
                                                    [0.05,0,0.05,0.9]])
        
    def probability(self, next_state, state, action):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, stays the same"""
        # attention
        normalized_prob = pdf_list(next_state.attention, self.sigma, self.attention_list)
        attention_trans_prob = normalized_prob[self.attention_list.index(next_state.attention)]
        #print("next_env_probability:", next_env_probability)
        
        # 現状態と次状態のインデックスを取得
        current_index = state.comfort * 2 + state.follow
        next_index = next_state.comfort * 2 + next_state.follow
        
        # othersの遷移確率
        others_trans_prob = self.others_trans_prob_table[current_index,next_index]

        # 総合的な遷移確率
        trans_prob = attention_trans_prob * others_trans_prob
        return trans_prob
            
    def sample(self, state, action):
        # attention
        normalized_prob = pdf_list(state.attention, self.sigma, self.state_list)
        next_attention = self.attention_list[torch.multinomial(normalized_prob, 1).item()]

        # 現状態と次状態のインデックスを取得
        current_index = state.comfort * 2 + state.follow
        sample = torch.multinomial(self.others_trans_prob_table[current_index], 1).item()
        
        # indexからcomfortとfollowを取得
        next_comfort = sample // 2
        next_follow = sample % 2
        
        # サンプリング結果
        return Model_State(next_attention, next_comfort, next_follow)
    
    def get_all_states(self):
        state_space = generate_state_space(self.attention_list, self.comforts, self.follows)
        STATES = [Model_State(s) for s in state_space]
        return STATES


class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        # 損失関数を定義
        return state.comfort

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)

class PolicyModel(pomdp_py.RolloutPolicy):  # 方策モデル
    def __init__(self,step=1,num_people=3):
        self.step = step
        self.n = num_people
        
        """A simple policy model with uniform prior over a
        small, finite action space"""

        self.weight_list = generate_weight_list(self.step,self.n)
        self.ACTIONS = [Model_Action(s) for s in self.weight_list]  # 行動の選択肢

    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        actions = generate_weight_list(self.step,self.n)
        ACTIONS = [Model_Action(a) for a in actions]
        return ACTIONS

class Model_Problem(pomdp_py.POMDP):
    """
    In fact, creating a Model_Problem class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, init_true_state, init_belief,step=1,num_people=3):
        self.step = step
        self.n = num_people
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(
            init_belief,
            PolicyModel(self.step,self.n),
            TransitionModel(self.step, self.n),
            ObservationModel(self.step,self.n),
            RewardModel(),
        )
        env = pomdp_py.Environment(init_true_state, TransitionModel(), RewardModel())
        super().__init__(agent, env, name="Model_Problem")

        @staticmethod
        def create(state=[0,1,0,1,1], belief=0.8):
            """
            Args:
                belief (float): Initial belief that the target is
                                on the left; Between 0-1.
                obs_noise (float): Noise for the observation
                                model (default 0.15)
            """
            init_true_state = Model_State(state)
            init_belief = pomdp_py.Histogram(
                {Model_Problem([0,1,0,1,1]) : belief, Model_Problem([0,1,0,0,1]) : 1-belief}
            )
            model_problem = Model_Problem(init_true_state, init_belief)
            model_problem.agent.set_belief(init_belief, prior=True)
            return model_problem
        
def test_planner(model_problem, planner, nsteps=3, debug_tree=False):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        model_problem (TigerProblem): a problem instance
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
        debug_tree (bool): True if get into the pdb with a
                           TreeDebugger created as 'dd' variable.
    """
    for i in range(nsteps):
        action = planner.plan(model_problem.agent)
        if debug_tree:
            from pomdp_py.utils import TreeDebugger

        print("==== Step %d ====" % (i + 1))
        print(f"True state: {model_problem.env.state}")
        print(f"Belief: {model_problem.agent.cur_belief}")
        print(f"Action: {action}")
        
        # actionを実行して報酬を得る
        #reward = model_problem.env.state_transition(action, execute=True)  # 行動によって状態が変化する場合
        reward = model_problem.env.reward_model.sample(
            model_problem.env.state, action, None
        )
        print("Reward:", reward)

        # 観測も得る
        real_observation = Model_Observation(model_problem.env.state.attention)
        print(">> Observation:", real_observation)
        model_problem.agent.update_history(action, real_observation)

        # Update the belief. If the planner is POMCP, planner.update　also automatically updates agent belief.
        # 信念の更新  # planner.updateはPOMCPの場合、エージェントの信念も自動的に更新する
        planner.update(model_problem.agent, action, real_observation)
        
        # Print some info about the planner
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims:", planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        # Print the tree
        if isinstance(model_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(
                model_problem.agent.cur_belief,
                action.tolist(),
                real_observation.tolist(),
                model_problem.agent.observation_model,
                model_problem.agent.transition_model,
            )
            model_problem.agent.set_belief(new_belief)

def make_model(init_state=[0,1,0,1,1], init_belief=[0.8,0.2],step=1,num_people=3):
    """model_domain の作成に便利"""
    model = Model_Problem(
        Model_State(init_state),
        pomdp_py.Histogram(
            {
                Model_State([0,1,0,1,1]): init_belief[0],
                Model_State([0,1,0,0,1]): init_belief[1],
            }
        ),
        step,
        num_people
    )
    return model


def main():
    init_true_state = [0,1,0,1,1]
    init_belief = pomdp_py.Histogram(
        {Model_State([0,1,0,1,1]): 0.8, Model_State([0,1,0,0,1]): 0.2}
    )
    # ハイパラの設定
    step = 1
    num_people = 3
    model = make_model(init_state=init_true_state,step=step,num_people=num_people)
    init_belief = model.agent.belief
    
    # 三つのプランナーを比較
    print("** Testing value iteration **")  # 価値反復法
    vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
    test_planner(model, vi, nsteps=10)
    """
    print("\n** Testing POUCT **")
    
    pouct = pomdp_py.POUCT(
        max_depth=3,
        discount_factor=0.95,
        num_sims=4096,
        exploration_const=50,
        rollout_policy=tiger.agent.policy_model,
        show_progress=True,
    )
    
    test_planner(tiger, pouct, nsteps=10)   # nsteps:学習回数
    TreeDebugger(tiger.agent.tree).pp

    # Reset agent belief
    tiger.agent.set_belief(init_belief, prior=True)
    tiger.agent.tree = None

    print("** Testing POMCP **")
    tiger.agent.set_belief(
        pomdp_py.Particles.from_histogram(init_belief, num_particles=100), prior=True
    )
    pomcp = pomdp_py.POMCP(
        max_depth=3,
        discount_factor=0.95,
        num_sims=1000,
        exploration_const=50,
        rollout_policy=tiger.agent.policy_model,
        show_progress=True,
        pbar_update_interval=500,
    )
    test_planner(tiger, pomcp, nsteps=10)
    TreeDebugger(tiger.agent.tree).pp
    """


if __name__ == "__main__":
    main()
"""
state = Model_State([0.4,0.3,0.3],[0.4,0.3,0.3])
next_state = Model_State([0.4,0.3,0.3],[0.4,0.3,0.3])
action = Model_Action([0.3,0.4,0.3])
observation = Model_Observation([0.3,0.4,0.3],[0.3,0.4,0.3])
obs_noise = 0.05
init_true_state = Model_State([0.4,0.3,0.3],[0.4,0.3,0.3])    # 初期状態
init_belief = pomdp_py.Histogram({Model_State([0.4,0.3,0.3],[0.4,0.3,0.3]):0.5, Model_State([0.3,0.4,0.3],[0.3,0.4,0.3]):0.5})  # 初期信念


observation_model = ObservationModel()
transition_model = TransitionModel()
reward_model = RewardModel()
policy_model = PolicyModel()
model_problem = Model_Problem(obs_noise, init_true_state, init_belief)
test_planner(model_problem, pomdp_py.POUCT(), nsteps=3, debug_tree=False)


#print("observation_probability :",observation_model.probability(observation, next_state, action))
#print("observation_sample", observation_model.sample(next_state, action))
#print("transition_probability :",transition_model.probability(next_state, state, action))
#print("transition_sample :", transition_model.sample(state, action))
#print("reward_sample :", reward_model._reward_func(state, action))
#print("policy_sample :", policy_model.sample(state))

model_problem = Model_Problem(obs_noise, init_true_state, init_belief)
print("model_problem :", model_problem)
print("model_problem.agent :", model_problem.agent)
model = make_model()
print("model :", model)
"""
