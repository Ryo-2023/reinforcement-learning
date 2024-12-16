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
    def __init__(self, environment, user):
        self.environment = environment
        self.user = user

    def __hash__(self):
        return hash((tuple(self.environment),tuple(self.user)))

    def __eq__(self, other):
        if isinstance(other, Model_State):
            return self.environment == other.environment and self.user == other.user
        return False

    def __str__(self):  # クラスの中身を表示
        return "speaking_activity :" + str(self.environment) + ", attention_weight" + str(self.user)

    def __repr__(self):
        return "Model_State(environment=%s, user=%s)" % (self.environment, self.user)
    
    """
    def other(self):
        if self.name.endswith("left"):
            return Model_State("tiger-right")
        else:
            return Model_State("tiger-left")
    """
        
class Model_Action(pomdp_py.Action):
    """
    Action = {
        "weight" : どのフィルタを重視するか 重みづけ
    }
    """
    def __init__(self, name):
        # Action は str型しか受けつけないことに注意
        if isinstance(name, list):
            self.name = str(name)
        else:
            self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Model_Action):
            return self.name == other.name
        return False

    def __str__(self):
        return self.name

    def __repr__(self):
        return "Model_Action(%s)" % self.name
    
class Model_Observation(pomdp_py.Observation):
    """
    Observation =
    {
        "watch_time_ratio"    : 視聴時間の比率  ex.) [0.15,0.7,0.15]
        "speaking_time_ratio" : 発話時間の比率
    }
    """
    def __init__(self, watch_time_ratio, speaking_time_ratio):
        self.watch_time_ratio = watch_time_ratio
        self.speaking_time_ratio = speaking_time_ratio

    def __hash__(self):
        return hash((tuple(self.watch_time_ratio), tuple(self.speaking_time_ratio)))

    def __eq__(self, other):
        if isinstance(other, Model_Observation):
            return self.watch_time_ratio == other.watch_time_ratio and self.speaking_time_ratio == other.speaking_time_ratio
        return False

    def __str__(self):
        return "watch_time_ratio :" + str(self.watch_time_ratio) + ", speaking_time_ratio" + str(self.speaking_time_ratio)

    def __repr__(self):
        return "Model_Observation(%s)" % self.name
    
def generate_weight_list():
    values = [round(i * 0.1, 2) for i in range(11)]
    return [[x,y,z] for x,y,z in itertools.product(values, repeat=3) if round(x+y+z,10)==1.0]

def generate_all_states():
    weight_list = generate_weight_list()
    all_states = []
    for env in weight_list:
        for user in weight_list:
            all_states.append(Model_State(env, user))
    return all_states

def pdf_list(mean, sigma, weight_list):
    probabilities = []
    for weights in weight_list:
        weights_tensor = torch.tensor(weights)
        diff = weights_tensor - mean.clone().detach()
        prob = (torch.exp(-0.5 * ((diff / sigma) ** 2))/(sigma*torch.sqrt(torch.tensor(2*torch.pi)))).prod().item()
        probabilities.append(prob)
    normalized_probabilities = [prob / sum(probabilities) for prob in probabilities]
    return normalized_probabilities
    
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, sigma=0.1,alpha=0.8,beta = 0.8):
        self.sigma = sigma # 分散
        self.alpha = alpha # 重み
        self.beta  = beta  # 重み
        self.obs_list = generate_weight_list()    # 観測のリストを生成

    def probability(self, observation, next_state, action,device):  # 観測確率の計算
        # watch_time_ratio
        watch_mean = self.alpha * torch.tensor(next_state.user) + (1 - self.alpha) * torch.tensor(ast.literal_eval(action.name))
        normalized_probabilities = pdf_list(watch_mean, self.sigma, self.obs_list)
        watch_probability = normalized_probabilities[self.obs_list.index(observation.watch_time_ratio)]  # 観測確率
        #print("watch_probability:", watch_probability)

        # speaking_time_ratio
        speak_mean = self.beta * torch.tensor(next_state.environment) + (1 - self.beta) * torch.tensor(ast.literal_eval(action.name))
        normalized_probabilities = pdf_list(speak_mean, self.sigma, self.obs_list)
        speaking_probability = normalized_probabilities[self.obs_list.index(observation.speaking_time_ratio)]  # 観測確率
        #print("speaking_probability:", speaking_probability)
        
        return [watch_probability, speaking_probability]

    def sample(self, next_state, action,device = "cpu"):  # 観測のサンプリング
        # watch_time_ratio
        watch_mean = self.alpha * torch.tensor(next_state.user) + (1 - self.alpha) * torch.tensor(ast.literal_eval(action.name))
        normalized_probabilities = pdf_list(watch_mean, self.sigma, self.obs_list)
        watch_sample = self.obs_list[torch.multinomial(torch.tensor(normalized_probabilities), 1).item()]    # torch.multinomial:与えられた確率分布に基づきサンプリング

        # speaking_time_ratio
        speak_mean = self.beta * torch.tensor(next_state.environment) + (1 - self.beta) * torch.tensor(ast.literal_eval(action.name))
        normalized_probabilities = pdf_list(speak_mean, self.sigma, self.obs_list) # 正規化
        speaking_sample = self.obs_list[torch.multinomial(torch.tensor(normalized_probabilities), 1).item()]

        return Model_Observation(watch_sample, speaking_sample)
        
    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space
        (e.g. value iteration)"""
        obs_list = generate_weight_list()
        all_observations = []
        for watch in obs_list:
            for speak in obs_list:
                all_observations.append(Model_Observation(watch,speak))
        return all_observations
    
class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self, sigma=0.1,alpha=0.8,beta=0.8):
        self.sigma = sigma # 分散
        self.alpha = alpha # 重み
        self.beta  = beta  # 重み
        self.state_list = generate_weight_list()    # 状態のリストを生成

    def probability(self, next_state, state, action, device):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, stays the same"""
        # environment
        env_mean = self.alpha * torch.tensor(state.environment).clone().detach() + (1 - self.alpha) * torch.tensor(ast.literal_eval(action.name)).clone().detach()
        normalized_probabilities = pdf_list(env_mean, self.sigma, self.state_list)
        next_env_probability = normalized_probabilities[self.state_list.index(next_state.environment)]
        #print("next_env_probability:", next_env_probability)
            
        # user
        user_mean = self.beta * torch.tensor(state.user).clone().detach() + (1 - self.beta) * torch.tensor(ast.literal_eval(action.name)).clone().detach()
        normalized_probabilities = pdf_list(user_mean, self.sigma, self.state_list)
        next_user_probability = normalized_probabilities[self.state_list.index(next_state.user)]
        #print("next_user_probability:", next_user_probability)
        
        return [next_env_probability, next_user_probability]
            
    def sample(self, state, action,device="cpu"):
        # environment
        env_mean = self.alpha * torch.tensor(state.environment) + (1 - self.alpha) * torch.tensor(ast.literal_eval(action.name))
        normalized_probabilities = pdf_list(env_mean, self.sigma, self.state_list)
        next_env_sample = self.state_list[torch.multinomial(torch.tensor(normalized_probabilities), 1).item()]

        # user
        user_mean = self.beta * torch.tensor(state.user) + (1 - self.beta) * torch.tensor(ast.literal_eval(action.name))
        normalized_probabilities = pdf_list(user_mean, self.sigma, self.state_list)
        next_user_sample = self.state_list[torch.multinomial(torch.tensor(normalized_probabilities), 1).item()]

        return Model_State(next_env_sample, next_user_sample)

    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)
        """
        state_list = generate_weight_list()
        all_states = []
        for env in state_list:
            for user in state_list:
                all_states.append(Model_State(env,user))
        return all_states

class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        # 損失関数を定義
        # ここでは、単純にMSE(平均二乗誤差)
        #print("state.user:", state)
        #print("action.name:",ast.literal_eval(action.name))
        mse = sklearn.metrics.mean_squared_error(state.user, ast.literal_eval(action.name))
        return -mse

    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)

class PolicyModel(pomdp_py.RolloutPolicy):  # 方策モデル
    """A simple policy model with uniform prior over a
    small, finite action space"""

    weight_list = generate_weight_list()
    ACTIONS = [Model_Action(s) for s in weight_list]  # 行動の選択肢

    def sample(self, state):
        return random.sample(self.get_all_actions(), 1)[0]

    def rollout(self, state, history=None):
        """Treating this PolicyModel as a rollout policy"""
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS

class Model_Problem(pomdp_py.POMDP):
    """
    In fact, creating a Model_Problem class is entirely optional
    to simulate and solve POMDPs. But this is just an example
    of how such a class can be created.
    """

    def __init__(self, obs_noise, init_true_state, init_belief):
        """init_belief is a Distribution."""
        agent = pomdp_py.Agent(
            init_belief,
            PolicyModel(),
            TransitionModel(),
            ObservationModel(obs_noise),
            RewardModel(),
        )
        env = pomdp_py.Environment(init_true_state, TransitionModel(), RewardModel())
        super().__init__(agent, env, name="Model_Problem")

        @staticmethod
        def create(state=[0.3,0.3,0.4], belief=0.5, obs_noise=0.05):
            """
            Args:
                belief (float): Initial belief that the target is
                                on the left; Between 0-1.
                obs_noise (float): Noise for the observation
                                model (default 0.15)
            """
            init_true_state = Model_State(state)
            all_states = generate_all_states()
            init_belief = {state : 0 for state in all_states}
            init_belief[Model_State([0.4,0.3,0.3],[0.4,0.3,0.3])] = belief
            init_belief[Model_State([0.3,0.3,0.4],[0.3,0.3,0.4])] = 1-belief
            init_belief = pomdp_py.Histogram(init_belief)

            model_problem = Model_Problem(obs_noise, init_true_state, init_belief)
            model_problem.agent.set_belief(init_belief, prior=True)
            return model_problem
        
class PolicyTreeNode:
    def __init__(self, device, action, depth, agent, discount_factor, children=None):
        if children is None:
            children = {}
        print("Initialize PolicyTreeNode")
        self.device = device
        self.action = action
        self.depth = depth
        self._agent = agent
        self.children = children
        self._discount_factor = discount_factor
        self.values = self._compute_values()  # s -> value
        
    def calculate_data_size(self,states, observations, actions):
        state_size = sum([torch.tensor(state.environment).element_size() * torch.tensor(state.environment).nelement() + 
                        torch.tensor(state.user).element_size() * torch.tensor(state.user).nelement() for state in states])
        observation_size = sum([torch.tensor(obs.watch_time_ratio).element_size() * torch.tensor(obs.watch_time_ratio).nelement() + 
                                torch.tensor(obs.speaking_time_ratio).element_size() * torch.tensor(obs.speaking_time_ratio).nelement() for obs in observations])
        action_size = sum([torch.tensor(ast.literal_eval(action.name)).element_size() * torch.tensor(ast.literal_eval(action.name)).nelement() for action in actions])
        total_size = state_size + observation_size + action_size
        return total_size
        
    def compute_future_values(self,args):
        s,sp,obs,action,agent,discount_factor,children,device = args
        expected_future_value = 0
        for o in tqdm(obs, desc="Processing observations (o)", leave=False):
            trans_prob = torch.tensor(agent.transition_model.probability(sp, s, action, device)).to(device)
            obsrv_prob = torch.tensor(agent.observation_model.probability(o, sp, action, device)).to(device)
            if len(children) > 0:
                subtree_value = children[o].values[sp]  # corresponds to V_{oi(p)} in paper
            else:
                subtree_value = 0.0
            reward = agent.reward_model.sample(s, action, sp)
            expected_future_value += torch.mean(trans_prob) * torch.mean(obsrv_prob) * (reward + discount_factor * subtree_value)
        return s, expected_future_value
    
    def calc_prob_GPU(args,sigma=0.1,gamma=0.8):
        self,device,s,o,action,obs_list = args
        mean = self.alpha * torch.tensor(s) + (1 - self.alpha) * torch.tensor(action)
        normalized_probabilities = pdf_list(mean, sigma, obs_list)
        probability = normalized_probabilities[self.obs_list.index(o)]  # 観測確率
        return probability
        
    def _compute_values(self):
        print("Computing values start")
        """
        Returns a dictionary {s -> value} that represents the values
        for the next actions.
        s : current state
        sp : next state
        o : observation
        """
        print("type_action:",type(self._agent.all_states))
        print("type_observation:",type(self._agent.all_observations))
        print("type_state:",type(self._agent.all_states))

        actions = self._agent.all_actions
        observations = self._agent.all_observations
        states = self._agent.all_states

        # torch.tensor に変換
        action_tensor = torch.tensor([ast.literal_eval(self.action.name) for a in actions], device=self.device)
        state_env_tensor = torch.tensor([state.environment for state in states], device=self.device)
        state_user_tensor = torch.tensor([state.user for state in states], device=self.device)
        obs_watch_tensor = torch.tensor([obs.watch_time_ratio for obs in observations], device=self.device)
        obs_speak_tensor = torch.tensor([obs.speaking_time_ratio for obs in observations], device=self.device)

        # 状態、行動、観測の全組み合わせを作成  [s.env, s.user, a, o.watch, o.speak]
        combi = list(itertools.product(state_env_tensor.tolist(), state_user_tensor.tolist(),action_tensor.tolist(), obs_watch_tensor.tolist(), obs_speak_tensor.tolist()))
        
        discount_factor = self._discount_factor ** self.depth  # 累積
        values = {}

        # デバイスの確認
        print("device:",self.device)

        # データ量を計算
        total_size = self.calculate_data_size(states, observations, actions)
        print(f"Total data size: {total_size / (1024 ** 2):.2f} MB")

        # メモリの閾値を設定（例：1GB）
        memory_threshold = 1 * 1024 ** 3

        if total_size < memory_threshold:
            print("Data size is within memory threshold. Proceeding with GPU.")
            # GPUを使用して並列計算
            num_gpus = torch.cuda.device_count()
            pool = mp.Pool(mp.cpu_count())
            manager = mp.Manager()
            queue = manager.Queue()
            results = []
            with tqdm(total=len(combi), desc="Processing combinations", leave=False) as pbar:
                args = [(s, sp, o, self.action, self._agent, discount_factor, self.children, i % num_gpus, queue) for i, (s, sp, a, o) in enumerate(combi)]
                for result in tqdm(pool.imap(self.compute_future_values, args), total=len(args), desc="Processing combinations", leave=False):
                    results.append(result)
                    while not queue.empty():
                        queue.get()
                        pbar.update(1)
            pool.close()
            pool.join()

            values = {s: expected_future_value for s, expected_future_value in results}
            """
            num_gpus = torch.cuda.device_count()  # GPUの数
            pool = mp.Pool(mp.cpu_count())  # CPUの数
            results = []
            with tqdm(total=len(states), desc="Processing states (s)", leave=False) as pbar_s:
                for s in states:
                    args = [(s, sp, observations, self.action, self._agent, discount_factor, self.children, i % num_gpus) for i, sp in enumerate(states)]
                    for results in tqdm(pool.imap(self.compute_future_values, args), total=len(states), desc="Processing next states (sp)", leave=False):
                        results.append(results)
                    #results.extend(pool.map(self.compute_future_values, args))
                    pbar_s.update(1)
            pool.close()
            pool.join()

            values = {s: expected_future_value for s, expected_future_value in results}
            """
        else:
            print("Data size exceeds memory threshold. Skipping parallel computation.")
            # 並列計算をスキップしてシングルスレッドで計算する場合の処理を追加
            with tqdm(total=len(states), desc="Processing states (s)", leave=False) as pbar_s:
                for s in states:
                    expected_future_value = 0
                    for sp in states:
                        for o in observations:
                            trans_prob = torch.tensor(self._agent.transition_model.probability(sp, s, self.action, self.device)).to(self.device)
                            obsrv_prob = torch.tensor(self._agent.observation_model.probability(o, sp, self.action, self.device)).to(self.device)
                            if len(self.children) > 0:
                                subtree_value = self.children[o].values[sp]  # corresponds to V_{oi(p)} in paper
                            else:
                                subtree_value = 0.0
                            reward = self._agent.reward_model.sample(s, self.action, sp)
                            expected_future_value += torch.mean(trans_prob) * torch.mean(obsrv_prob) * (reward + discount_factor * subtree_value)
                    values[s] = expected_future_value
                    pbar_s.update(1)
        return values

        values = {s: expected_future_value for s, expected_future_value in results}
        return values
        """
        with tqdm(total=len(states), desc="Processing states (s)",leave=False) as pbar_s:
            for s in states:
                expected_future_value = 0
                #print("s:", s)
                # 中間のループ (sp) の進行状況バー
                with tqdm(total=len(states), desc="Processing next states (sp)", leave=False) as pbar_sp:
                    for sp in states:
                        #print("sp:", sp)
                        # 内側のループ (o) の進行状況バー
                        with tqdm(total=len(observations), desc="Processing observations (o)", leave=False) as pbar_o:
                            for o in observations:
                                #print("o:", o)
                                trans_prob = torch.tensor(self._agent.transition_model.probability(sp, s, self.action, self.device)).to(self.device)
                                obsrv_prob = torch.tensor(self._agent.observation_model.probability(o, sp, self.action, self.device)).to(self.device)
                                #print("self.children: ", self.children)
                                if len(self.children) > 0:
                                    subtree_value = self.children[o].values[sp]  # corresponds to V_{oi(p)} in paper
                                else:
                                    subtree_value = 0.0
                                reward = self._agent.reward_model.sample(s, self.action, sp)
                                #print("reward: ", reward)
                                expected_future_value += torch.mean(trans_prob) * torch.mean(obsrv_prob) * (reward + discount_factor * subtree_value)
                                pbar_o.update(1)  # 内側のループの進行状況バーを更新
                        pbar_sp.update(1)  # 中間のループの進行状況バーを更新
                pbar_s.update(1)
        """
        
        #return values

    def __str__(self):
        return "_PolicyTreeNode(%s, %d){%s}" % (self.action, self.depth, str(self.children.keys()))

    def __repr__(self):
        return self.__str__()

class ValueIteration(Planner):
    """
    This algorithm is only feasible for small problems where states, actions,
    and observations can be explicitly enumerated.

    __init__(self, horizon=float('inf'), discount_factor=0.9, epsilon=1e-6)
    """
    def __init__(self, device, horizon, discount_factor=0.9, epsilon=1e-6):
        """
        The horizon satisfies discount_factor**horizon > epsilon"""
        assert isinstance(horizon, int) and horizon >= 1, "Horizon must be an integer >= 1"
        self.device = device
        self._discount_factor = discount_factor
        self._epsilon = epsilon
        self._planning_horizon = horizon

    def plan(self, agent):
        print("plan start")
        """plan(self, agent)
        Plans an action."""
        policy_trees = self._build_policy_trees(1, agent)
        value_beliefs = {}
        for p, policy_tree in enumerate(policy_trees):
            value_beliefs[p] = 0
            for state in agent.all_states:
                value_beliefs[p] += agent.cur_belief[state] * policy_tree.values[state]
        # Pick the policy tree with highest belief value
        pmax = max(value_beliefs, key=value_beliefs.get)
        return policy_trees[pmax].action

    def _build_policy_trees(self, depth, agent):
        print("Build policy trees start")
        """Bottom up build policy trees"""
        actions = agent.all_actions
        states = agent.all_states
        observations = agent.all_observations  # we expect observations to be indexed

        if depth >= self._planning_horizon or self._discount_factor ** depth < self._epsilon:
            return [PolicyTreeNode(self.device, a, depth, agent, self._discount_factor) for a in actions]
        else:
            # Every observation can lead to K possible sub policy trees, which
            # is exactly the output of _build_policy_trees. Then, for a set of
            # observations, a policy tree is formed by combining together one
            # sub policy tree (corresponding to one action in the next time
            # step) per observation from the pool of K possible sub policy
            # trees. So we take the cartesian product of these sets of sub
            # policy trees and build individual policy trees.
            groups = [self._build_policy_trees(depth + 1, agent) for i in range(len(observations))]
            # (Sanity check) We expect all groups to have same size
            group_size = len(groups[0])
            for g in groups:
                assert group_size == len(g)

            # This computes all combinations of sub policy trees. Each combination
            # will become one policy tree that will be returned, with an action to
            # take at the current depth level as the root.
            combinations = itertools.product(*([np.arange(group_size)] * len(observations)))
            policy_trees = []
            for comb in combinations:
                # comb is a tuple of indicies, e.g. (i, j, k) that means
                # for observation 0, the sub policy tree is at index i of its group;
                children = {observations[i]: groups[i][comb[i]] for i in range(len(observations))}
                for a in actions:
                    policy_trees.append(PolicyTreeNode(self.device, a, depth, agent, self._discount_factor, children))
            return policy_trees

def test_planner(model_problem, planner, nsteps=10, debug_tree=True):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        model_problem (Model_Problem): a problem instance
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
        debug_tree (bool): True if get into the pdb with a
                           TreeDebugger created as 'dd' variable.
    """
    for i in range(nsteps):
        action = planner.plan(model_problem.agent)  # 各ステップにおいて、方策に従い行動を選択

        print("==== Step %d ====" % (i + 1))
        print(f"True state: {model_problem.env.state}")
        print(f"Belief: {model_problem.agent.cur_belief}")
        print(f"Action: {action}")
        # There is no state transition for the tiger domain.
        # In general, the ennvironment state can be transitioned
        # using
        #
        #   reward = model_problem.env.state_transition(action, execute=True)
        #
        # Or, it is possible that you don't have control
        # over the environment change (e.g. robot acting
        # in real world); In that case, you could skip
        # the state transition and re-estimate the state
        # (e.g. through the perception stack on the robot).
        reward = model_problem.env.reward_model.sample(
            model_problem.env.state, action, None
        )
        print("Reward:", reward)

        # Let's create some simulated real observation;
        # Here, we use observation based on true state for sanity
        # checking solver behavior. In general, this observation
        # should be sampled from agent's observation model, as
        #
        #    real_observation = model_problem.agent.observation_model.sample(model_problem.env.state, action)
        #
        # or coming from an external source (e.g. robot sensor
        # reading). Note that model_problem.env.state stores the
        # environment state after action execution.
        real_observation = Model_Observation(model_problem.env.state.name)
        print(">> Observation:", real_observation)
        model_problem.agent.update_history(action, real_observation)

        # Update the belief. If the planner is POMCP, planner.update
        # also automatically updates agent belief.
        planner.update(model_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims:", planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        if isinstance(model_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(
                model_problem.agent.cur_belief,
                action,
                real_observation,
                model_problem.agent.observation_model,
                model_problem.agent.transition_model,
            )
            model_problem.agent.set_belief(new_belief)

        print("\n")

def make_model(init_env,init_user,noise=0.05):
    """Convenient function to quickly build a model domain.
    Useful for testing"""
    weight_list = generate_weight_list()
    all_states = generate_all_states()
    init_belief = {state : 0 for state in all_states}
    init_belief[Model_State([0.4,0.3,0.3],[0.4,0.3,0.3])] = 0.5
    init_belief[Model_State([0.3,0.3,0.4],[0.3,0.3,0.4])] = 0.5
    model = Model_Problem(
        noise,
        Model_State(init_env,init_user),
        pomdp_py.Histogram(init_belief),
    )
    return model

def main():
    init_state_env = [0.4,0.3,0.3]
    init_state_user = [0.4,0.3,0.3]
    init_belief = pomdp_py.Histogram(
        {Model_State([0.4,0.33,0.3],[0.3,0.4,0.3]):0.5,
        Model_State([0.3,0.3,0.4],[0.3,0.3,0.4]):0.5}
    )
    model = make_model(init_env = init_state_env, init_user = init_state_user)
    init_belief = model.agent.belief
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # デバイスの設定

    # 三つのプランナーを比較
    print("** Testing value iteration **")  # 価値反復法
    vi = ValueIteration(device, horizon=3, discount_factor=0.95, )
    test_planner(model, vi, nsteps=10)

    print("\n** Testing POUCT **")

    pouct = pomdp_py.POUCT(
        max_depth=3,
        discount_factor=0.95,
        num_sims=4096,
        exploration_const=50,
        rollout_policy=model.agent.policy_model,
        show_progress=True,
    )

    test_planner(model, pouct, nsteps=10)   # nsteps:学習回数
    TreeDebugger(model.agent.tree).pp

    # Reset agent belief
    model.agent.set_belief(init_belief, prior=True)
    model.agent.tree = None

    print("** Testing POMCP **")
    model.agent.set_belief(
        pomdp_py.Particles.from_histogram(init_belief, num_particles=100), prior=True
    )
    pomcp = pomdp_py.POMCP(
        max_depth=3,
        discount_factor=0.95,
        num_sims=1000,
        exploration_const=50,
        rollout_policy=model.agent.policy_model,
        show_progress=True,
        pbar_update_interval=500,
    )
    test_planner(model, pomcp, nsteps=1000)
    TreeDebugger(model.agent.tree).pp


if __name__ == "__main__":
    main()