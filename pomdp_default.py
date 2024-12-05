import pomdp_py
from pomdp_py.utils import TreeDebugger
import random
import torch
import sys
import copy
import scipy
import sklearn
import itertools
import ast

"""
各クラスは単なるコンテナであり、具体的な状態は保持しない
"""
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
        return hash((self.environment,self.user))

    def __eq__(self, other):
        if isinstance(other, Model_State):
            return self.environment == other.environment and self.user == other.user
        return False

    def __str__(self):  # クラスの中身を表示
        return "speaking_activity :" + str(self.environment) + ", attention_weight" + str(self.user)

    def __repr__(self):
        return "Model_State(%s)" % self.name
    
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
        return hash((self.watch_time_ratio, self.speaking_time_ratio))

    def __eq__(self, other):
        if isinstance(other, Model_Observation):
            return self.watch_time_ratio == other.watch_time_ratio and self.speaking_time_ratio == other.speaking_time_ratio
        return False

    def __str__(self):
        return self.watch_time_ratio + " " + self.speaking_time_ratio

    def __repr__(self):
        return "Model_Observation(%s)" % self.name
    
class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, sigma=0.25,alpha=0.8,beta = 0.7):
        self.sigma = sigma # 分散
        self.alpha = alpha # 重み
        self.beta  = beta  # 重み
        
    def probability(self, observation, next_state, action): # 観測確率の計算
        # watch_time_ratio 
        watch_mean = [self.alpha * x for x in next_state.user] + [(1-self.alpha) * y for y in action.name]
        watch_probability = []
        for i in range(len(watch_mean)):
            p1 = scipy.stats.norm.pdf(observation.watch_time_ratio, watch_mean[i], self.sigma)
            watch_probability[i] = p1
        # speaking_time_ratio 
        speaking_mean = [self.beta * x for x in next_state.environment] + [(1-self.beta) * y for y in action.name]
        speaking_probability = []
        for i in range(len(speaking_mean)):
            p2 = scipy.stats.norm.pdf(observation.speaking_time_ratio, speaking_mean[i], self.sigma)
            speaking_probability[i] = p2
        return [watch_probability, speaking_probability]

    def sample(self, next_state, action):   # 観測のサンプリング
        # watch_time_ratio
        mean1 = [self.alpha * x for x in next_state.user] + [(1-self.alpha) * y for y in action.name]
        watch_sample = []
        for i in range(len(mean1)):
            p1 = scipy.stats.norm.pdf(next_state.watch_time_ratio, mean1[i], self.sigma)
            watch_sample.append(p1)
        # speaking_time_ratio
        mean2 = [self.beta * x for x in next_state.environment] + [(1-self.beta) * y for y in action.name]
        speaking_sample = []
        for i in range(len(mean2)):
            p2 = scipy.stats.norm.pdf(next_state.environment, mean2[i], self.sigma)
            speaking_sample.append(p2)
        return Model_Observation(watch_sample, speaking_sample)
    
    def get_all_observations(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space
        (e.g. value iteration)"""
        return [Model_Observation(s) for s in {"tiger-left", "tiger-right"}]
    
class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self, sigma=0.25,alpha=0.8,beta=0.8):
        self.sigma = sigma # 分散
        self.alpha = alpha # 重み
        self.beta  = beta  # 重み
        
    def probability(self, next_state, state, action):
        """According to problem spec, the world resets once
        action is open-left/open-right. Otherwise, stays the same"""
        # speaking_activity
        mean_a = self.alpha * state.speaking_time_ratio + (1-self.alpha) * action.name
        next_speaking_activity = []
        for i in range(len(mean_a)):
            p1 = scipy.stats.norm.pdf(next_state.environment, mean_a[i], self.sigma)
            next_speaking_activity.append(p1)
        # attention_weight
        mean_b = self.beta * state.attention_weight + (1-self.beta) * action.name
        next_attention_weight = []
        for i in range(len(mean_b)):
            p2 = scipy.stats.norm.pdf(next_state.user, mean_b[i], self.sigma)
            next_attention_weight.append(p2)
        return [next_speaking_activity, next_attention_weight]
            
    def sample(self, state, action):
        # speaking_activity
        mean_a = self.alpha * state.speaking_time_ratio + (1-self.alpha) * action.name
        speaking_activity = []
        for i in range(len(mean_a)):
            p1 = scipy.stats.norm.pdf(state.speaking_time_ratio, mean_a[i], self.sigma)
            speaking_activity.append(p1)
        # attention_weight
        mean_b = self.beta * state.attention_weight + (1-self.beta) * action.name
        attention_weight = []
        for i in range(len(mean_b)):
            p2 = scipy.stats.norm.pdf(state.attention_weight, mean_b[i], self.sigma)
            attention_weight.append(p2)
        return Model_State(speaking_activity, attention_weight) # 次の状態のサンプリング
        
    def get_all_states(self):
        """Only need to implement this if you're using
        a solver that needs to enumerate over the observation space (e.g. value iteration)
        """
        return [Model_State(s) for s in {"tiger-left", "tiger-right"}]

class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        # 損失関数を定義
        # ここでは、単純にMSE(平均二乗誤差)
        mse = sklearn.matrics.mean_squared_error(state,action)
        return -mse
        
    def sample(self, state, action, next_state):
        # deterministic
        return self._reward_func(state, action)
    
class PolicyModel(pomdp_py.RolloutPolicy):  # 方策モデル
    """A simple policy model with uniform prior over a
    small, finite action space"""

    values = [round(i * 0.05,2) for i in range(21)]
    weight_list = [
        [x,y,z] for x,y,z in itertools.product(values, repeat=3) if round(x+y+z,10)==1.0
    ]
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
        def create(state="[0,1,0]", belief=0.5, obs_noise=0.25):
            """
            Args:
                state (str): could be 'tiger-left' or 'tiger-right';
                            True state of the environment
                belief (float): Initial belief that the target is
                                on the left; Between 0-1.
                obs_noise (float): Noise for the observation
                                model (default 0.15)
            """
            init_true_state = Model_State(state)
            init_belief = pomdp_py.Histogram(
                {Model_State("tiger-left"): belief, Model_State("tiger-right"): 1.0 - belief}
            )
            model_problem = Model_Problem(obs_noise, init_true_state, init_belief)
            model_problem.agent.set_belief(init_belief, prior=True)
            return model_problem
        
def test_planner(tiger_problem, planner, nsteps=3, debug_tree=False):
    """
    Runs the action-feedback loop of Tiger problem POMDP

    Args:
        tiger_problem (Model_Problem): a problem instance
        planner (Planner): a planner
        nsteps (int): Maximum number of steps to run this loop.
        debug_tree (bool): True if get into the pdb with a
                           TreeDebugger created as 'dd' variable.
    """
    for i in range(nsteps):
        action = planner.plan(tiger_problem.agent)
        if debug_tree:
            from pomdp_py.utils import TreeDebugger

        print("==== Step %d ====" % (i + 1))
        print(f"True state: {tiger_problem.env.state}")
        print(f"Belief: {tiger_problem.agent.cur_belief}")
        print(f"Action: {action}")
        # There is no state transition for the tiger domain.
        # In general, the ennvironment state can be transitioned
        # using
        #
        #   reward = tiger_problem.env.state_transition(action, execute=True)
        #
        # Or, it is possible that you don't have control
        # over the environment change (e.g. robot acting
        # in real world); In that case, you could skip
        # the state transition and re-estimate the state
        # (e.g. through the perception stack on the robot).
        reward = tiger_problem.env.reward_model.sample(
            tiger_problem.env.state, action, None
        )
        print("Reward:", reward)

        # Let's create some simulated real observation;
        # Here, we use observation based on true state for sanity
        # checking solver behavior. In general, this observation
        # should be sampled from agent's observation model, as
        #
        #    real_observation = tiger_problem.agent.observation_model.sample(tiger_problem.env.state, action)
        #
        # or coming from an external source (e.g. robot sensor
        # reading). Note that tiger_problem.env.state stores the
        # environment state after action execution.
        real_observation = Model_Observation(tiger_problem.env.state.name)
        print(">> Observation:", real_observation)
        tiger_problem.agent.update_history(action, real_observation)

        # Update the belief. If the planner is POMCP, planner.update
        # also automatically updates agent belief.
        planner.update(tiger_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims:", planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

        if isinstance(tiger_problem.agent.cur_belief, pomdp_py.Histogram):
            new_belief = pomdp_py.update_histogram_belief(
                tiger_problem.agent.cur_belief,
                action,
                real_observation,
                tiger_problem.agent.observation_model,
                tiger_problem.agent.transition_model,
            )
            tiger_problem.agent.set_belief(new_belief)

        if action.name.startswith("open"):
            # Make it clearer to see what actions are taken
            # until every time door is opened.
            print("\n")

def make_tiger(noise=0.15, init_state="tiger-left", init_belief=[0.5, 0.5]):
    """Convenient function to quickly build a tiger domain.
    Useful for testing"""
    tiger = Model_Problem(
        noise,
        Model_State(init_state),
        pomdp_py.Histogram(
            {
                Model_State("tiger-left"): init_belief[0],
                Model_State("tiger-right"): init_belief[1],
            }
        ),
    )
    return tiger


def main():
    init_true_state = random.choice(["tiger-left", "tiger-right"])
    init_belief = pomdp_py.Histogram(
        {Model_State("tiger-left"): 0.5, Model_State("tiger-right"): 0.5}
    )
    tiger = make_tiger(init_state=init_true_state)
    init_belief = tiger.agent.belief
    
    # 三つのプランナーを比較
    print("** Testing value iteration **")  # 価値反復法
    vi = pomdp_py.ValueIteration(horizon=3, discount_factor=0.95)
    test_planner(tiger, vi, nsteps=10)

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
    test_planner(tiger, pomcp, nsteps=1000)
    TreeDebugger(tiger.agent.tree).pp


if __name__ == "__main__":
    main()