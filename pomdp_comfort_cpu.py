import pomdp_py
from pomdp_py.utils import TreeDebugger
from value_iteration import ValueIteration
from PBVI import PBVI
import random
import torch
import itertools
from tqdm import tqdm
from pomdp_py.utils import TreeDebugger
import pickle
import torch.nn.functional as F
import time
import os

"""
各クラスの詳細 : pomdp_py.framework.basics
update belief : pomdp_py.representations.belief.histogram
"""

class Model_State(pomdp_py.State):
    """
    state =
    {
        "attention": ユーザーの注意の向き方 
        "comfort"  : 快適度
        "follow"   : システムに従うかどうか
        "state_space": 状態空間
        states: ([[0., 0., 1., 0., 0.],
                 [0., 0., 1., 0., 1.],
                 [0., 0., 1., 1., 0.],
                 [0., 0., 1., 1., 1.],
                 [0., 1., 0., 0., 0.],
                 [0., 1., 0., 0., 1.],
                 [0., 1., 0., 1., 0.],
                 [0., 1., 0., 1., 1.],
                 [1., 0., 0., 0., 0.],
                 [1., 0., 0., 0., 1.],
                 [1., 0., 0., 1., 0.],
                 [1., 0., 0., 1., 1.]]) shape:(12, 5)
        } 
    }
    """
    def __init__(self, state = [0,1,0,1,1],step=1,n_people=3):
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
        self.states = generate_state_space(self.attentions, self.comforts, self.follows)

    def __hash__(self):
        #print("type(self.state):",type(tuple(self.state)))
        return hash((tuple(self.attention.tolist()), self.comfort, self.follow))

    def __eq__(self, other):
        if not isinstance(other, Model_State):
            return False
        else:
            return (torch.equal(self.attention,other.attention) and 
                                self.comfort == other.comfort and 
                                self.follow == other.follow)

    def __str__(self):  # クラスの中身を表示
        return "Model_State(attention=%s, comfort=%s, follow = %s)" % (self.attention.tolist(), self.comfort, self.follow)

    def __repr__(self):
        return "Model_State(attention=%s, comfort=%s, follow = %s)" % (self.attention.tolist(), self.comfort, self.follow)
    
    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the state space (e.g. value iteration)
        """
        state_list =  generate_state_space(self.attentions, self.comforts, self.follows)
        state_space = [Model_State(s) for s in state_list]
        return state_space
    
    def num_states(self):
        return len(self.get_all_states())

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
        self.step = step   # 行動空間の生成

    def __hash__(self):
        return hash(self.enhance_weight)

    def __eq__(self, other):
        if isinstance(other, Model_Action):
            return self.enhance_weight == other.enhance_weight
        return False

    def __str__(self):
        return "enhance_weight : %s" % self.enhance_weight.tolist()

    def __repr__(self):
        return "Model_Action(%s)" % self.enhance_weight
    
    def get_all_actions(self):
        action_list = generate_weight_list(step=self.step)
        ACTIONS = [Model_Action(a) for a in action_list]
        return ACTIONS
          
    def num_actions(self):
        return len(self.get_all_actions())
    
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
        self.obs_space = generate_weight_list(self.step,self.n)    # 観測空間の生成

    def __hash__(self):
        return hash(tuple(self.sight_direction.tolist()))

    def __eq__(self, other):
        if isinstance(other, Model_Observation):
            return torch.equal(self.sight_direction , other.sight_direction)
        return False

    def __str__(self):
        return "sight_direction:" + str(self.sight_direction)

    def __repr__(self):
        return "Model_Observation(%s)" % self.sight_direction.tolist()
    
    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space
        (e.g. value iteration)"""
        all_observations = []
        all_obs = generate_weight_list(self.step,self.n)
        for obs in all_obs:
            all_observations.append(Model_Observation(obs))
        return all_observations
    
    def get_num_observations(self):
        return len(self.get_all_observations())
    
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
    def __init__(self,step = 1,n_people=3, sigma=0.3):
        self.sigma = sigma # 分散
        self.step = step
        self.n = n_people
        self.obs_list = generate_weight_list(self.step,self.n)    # 観測のリストを生成

    def probability(self, obs, next_state, action):  # follow = 1
        if next_state.follow == 1:
            normalized_prob = pdf_list(action.enhance_weight, self.sigma, self.obs_list)
            sight_prob_follow = normalized_prob[self.obs_list.tolist().index(obs.sight_direction.tolist())]  # 観測確率   # .index使うためにリストに変換
            return sight_prob_follow
        elif next_state.follow == 0:
            normalized_prob = pdf_list(next_state.attention, self.sigma, self.obs_list)
            sight_prob_not_follow = normalized_prob[self.obs_list.tolist().index(obs.sight_direction.tolist())]  # 観測確率
            return sight_prob_not_follow

    def sample(self, next_state, action):  # 観測のサンプリング, 実データで学習する場合は、ここに実データを入れる?
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
    def __init__(self, step=1,n_people=3, sigma=0.3):
        self.sigma = sigma # 分散
        self.step = step
        self.n = n_people
        
        # 状態空間の生成
        self.attention_list = generate_weight_list(self.step,self.n)
        self.comforts = torch.tensor([0,1])
        self.follows  = torch.tensor([0,1])
        self.others_list = torch.tensor([[0,0],[0,1],[1,0],[1,1]])
        self.others_trans_prob_table = torch.tensor([[0.7,0.2,0.05,0.05],
                                                    [0.15,0.7,0.05,0.1],
                                                    [0.05,0.05,0.7,0.2],
                                                    [0.05,0,0.35,0.6]])
        
    def probability(self, next_state, state, action):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, stays the same"""
        # attention
        normalized_prob = pdf_list(next_state.attention, self.sigma, self.attention_list)
        attention_trans_prob = normalized_prob[self.attention_list.tolist().index(next_state.attention.tolist())]
        #print("next_env_probability:", next_env_probability)
        
        # 現状態と次状態のインデックスを取得
        current_index = int(state.comfort * 2 + state.follow)
        next_index = int(next_state.comfort * 2 + next_state.follow)
        
        # othersの遷移確率
        others_trans_prob = self.others_trans_prob_table[current_index,next_index]

        # 総合的な遷移確率
        trans_prob = attention_trans_prob * others_trans_prob
        #print(f"trans_prob at ({state} -> {next_state}): {trans_prob:.4f}")
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
        return state.comfort-1.5*(F.mse_loss(state.attention, action.enhance_weight)).item()
        #return (F.mse_loss(state.attention, action.enhance_weight)).item()

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)

class PolicyModel(pomdp_py.RolloutPolicy):  # 方策モデル
    def __init__(self,step=1,num_people=3):
        self.step = step
        self.n = num_people
        
        """A simple policy model with uniform prior over a small, finite action space"""

        self.weight_list = generate_weight_list(self.step,self.n)
        self.ACTIONS = [Model_Action(s) for s in self.weight_list]  # 行動の選択肢

    def sample(self, state):
        # action はランダムにサンプリング
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

        """
        @staticmethod
        def create(state=[0,1,0,1,1], belief=0.8):
            
            Args:
                belief (float): Initial belief that the target is
                                on the left; Between 0-1.
                obs_noise (float): Noise for the observation
                                model (default 0.15)
            
            init_true_state = Model_State(state)
            belief = 0.8
            init_belief = pomdp_py.Histogram(
                {Model_Problem([0,1,0,1,1]) : belief, Model_Problem([0,1,0,0,1]) : 1-belief},
            )
            model_problem = Model_Problem(init_true_state, init_belief)
            model_problem.agent.set_belief(init_belief, prior=True)
            return model_problem
        """
        
def test_planner(model_problem, planner, obs_data = None, nsteps=3, debug_tree=True,file_name_state = None, file_name_belief = None, file_name_action = None):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        model_problem : Model_Problem instance
        planner (Planner): planner
        nsteps (int) : 最大ステップ数
        debug_tree (bool): True if get into the pdb with a
                           TreeDebugger created as 'dd' variable.
    """
    # データリストの初期化
    data_state = []
    data_belief = []
    data_action = []
    
    for i in range(nsteps):
        start_time = time.time()
        action = planner.plan(model_problem.agent)
        if debug_tree:
            from pomdp_py.utils import TreeDebugger

        print("==== Step %d ====" % (i + 1))
        print(f"True state: {model_problem.env.state}")
        print(f"Belief: {model_problem.agent.cur_belief}")
        print(f"Action: {action}")
        
        # Save data
        
        data_state.append(model_problem.env.state.state.tolist())
        data_action.append(action.enhance_weight.tolist())
        
        # 保存用にデータ形式の変換
        convert_belief = {}
        for s,p in model_problem.agent.cur_belief.histogram.items():
            convert_belief[str(s.state.tolist())] = float(p)
        data_belief.append(convert_belief)
        
        # actionを実行して報酬を得る
        #reward = model_problem.env.state_transition(action, execute=True)  # 行動によって状態が変化する場合
        reward = model_problem.env.reward_model.sample(
            model_problem.env.state, action, None
        )
        print("Reward:", reward)

        # 観測のサンプリング
        # 入力の観測データがある場合は、それを使い、ない場合は観測モデルからサンプリング
        if obs_data is not None:
            real_observation = obs_data[i]
        else:
            real_observation = model_problem.env.observation_model.sample(model_problem.env.state, action)
        print(">> Observation:", real_observation)
        model_problem.agent.update_history(action, real_observation)

        # 信念の更新  # planner.updateはPOMCPの場合、エージェントの信念も自動的に更新する
        # 現在は後の update_histogram_belief で信念を更新しているため、いらない
        #planner.update(model_problem.agent, action, real_observation)
        
        # Print some info about the planner
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims:", planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        # Print the tree
        # update belief
        if isinstance(model_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(
                model_problem.agent.cur_belief,
                action,
                real_observation,
                model_problem.agent.observation_model,
                model_problem.agent.transition_model,
            )
            model_problem.agent.set_belief(new_belief)
            
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f"epoch_time:{epoch_time:.4f} s")
    
    # データの保存
    os.makedirs(os.path.dirname(file_name_state), exist_ok=True)  # 指定した保存先のディレクトリがない場合、自動生成
    if file_name_state is not None:
        save_data(data_state,file_name_state)
    if file_name_belief is not None:
        save_data(data_belief,file_name_belief)
    if file_name_action is not None:
        save_data(data_action,file_name_action)
    

def make_model(init_state=[0,1,0,1,1], init_belief=None,step=1,num_people=3):
    """model_domain の作成に便利"""
    # 初期信念の設定
    # 一様分布で信念を初期化
    all_states = Model_State().get_all_states()
    uniform_prob = 1.0 / len(all_states)
    belief_dict = {s : uniform_prob for s in all_states}
    
    model = Model_Problem(
        Model_State(init_state),
        pomdp_py.Histogram(belief_dict) if init_belief is None else pomdp_py.Histogram(init_belief),  # 初期信念
        step,
        num_people
    )
    return model

def save_data(data,file_name):
    with open(file_name, "wb") as f:
        pickle.dump(data, f)

def main():
    # Set default device and dtype
    device = "cpu"
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    
    # set save file name
    file_name_state = "E:/sotsuron/venv_sotsuron/src/VI_data/hybrid/data_state.pkl"
    file_name_belief = "E:/sotsuron/venv_sotsuron/src/VI_data/hybrid/data_belief.pkl"
    file_name_action = "E:/sotsuron/venv_sotsuron/src/VI_data/hybrid/data_action.pkl"
    
    # 初期状態の設定
    init_true_state = [0,1,0,1,1]
    
    # 初期信念の設定
    # 一様分布で信念を初期化
    all_states = Model_State().get_all_states()
    uniform_prob = 1.0 / len(all_states)
    belief_dict = {s : uniform_prob for s in all_states}
    
    # 特定の状態の信念を変更
    belief_dict[Model_State([0,1,0,1,1])] = 0.7
    belief_dict[Model_State([0,1,0,0,1])] = 0.2
    
    # 確率の正規化
    total_prob = sum(belief_dict.values())
    for p in belief_dict:
        belief_dict[p] /= total_prob
        
    # 初期信念の設定
    init_belief = pomdp_py.Histogram(belief_dict)
    
    # 観測データ
    file_path = "E:/sotsuron/venv_sotsuron/src/test_data/test_data.pkl"
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    obs_data = [Model_Observation(d) for d in data]
    
    # ハイパラの設定
    step = 1
    num_people = 3
    model = make_model(init_state=init_true_state,step=step,num_people=num_people)
    init_belief = model.agent.belief
    nsteps = len(obs_data) if obs_data is not None else 10
    
    # 三つのプランナーを比較
    # 価値反復法
    
    print("** Testing value iteration **")
    vi = ValueIteration(horizon=2, discount_factor=0.9)  # horizon:探索深度, horizon=3 にすると計算量が発散するためやめましょう
    all_start_time = time.time()
    print("start test_planner")
    test_planner(model, vi, obs_data, nsteps,None,file_name_state,file_name_belief,file_name_action)  # nsteps:学習回数
    all_end_time = time.time()
    print(f"all_time:{all_end_time - all_start_time:.4f} s")
    

if __name__ == "__main__":
    main()
