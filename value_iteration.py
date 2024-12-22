from pomdp_py.framework.planner import Planner
from pomdp_py.framework.basics import Agent, Action, State
import torch
import itertools

class _PolicyTreeNode:
    def __init__(self, action, depth, agent,
                 discount_factor, children={}):
        self.action = action
        self.depth = depth 
        self._agent = agent
        self.children = children
        self._discount_factor = discount_factor
        self.values = self._compute_values()  # s -> value

    def _compute_values(self):
        """
        Returns a dictionary {s -> value} that represents the values
        for the next actions.
        """
        actions = self._agent.all_actions
        observations = self._agent.all_observations
        states = self._agent.all_states

        discount_factor = self._discount_factor**self.depth
        values = {}
        for s in states:
            expected_future_value = 0.0
            for sp in states:
                for o in observations:
                    trans_prob = self._agent.transition_model.probability(sp, s, self.action)
                    obsrv_prob = self._agent.observation_model.probability(o, sp, self.action)
                    if len(self.children) > 0:
                        subtree_value = self.children[o].values[sp]  # corresponds to V_{oi(p)} in paper
                    else:
                        subtree_value = 0.0
                    reward = self._agent.reward_model.sample(s, self.action, sp)
                    expected_future_value += trans_prob * obsrv_prob * (reward + discount_factor*subtree_value)
            values[s] = expected_future_value
        return values

    def __str__(self):
        return "_PolicyTreeNode(%s, %d){%s}" % (self.action, self.depth, str(self.children_keys))
    def __repr__(self):
        return self.__str__()


class ValueIteration(Planner):
    """
    This algorithm is only feasible for small problems where states, actions,
    and observations can be explicitly enumerated.

    __init__(self, horizon=float('inf'), discount_factor=0.9, epsilon=1e-6)
    """

    def __init__(self, horizon, discount_factor=0.9, epsilon=1e-6):
        """
        The horizon satisfies discount_factor**horizon > epsilon"""
        assert type(horizon) == int and horizon >= 1, "Horizon must be an integer >= 1"
        self._discount_factor = discount_factor
        self._epsilon = epsilon
        self._planning_horizon = horizon

    def plan(self, agent):
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
        """Bottom up build policy trees"""
        actions = agent.all_actions
        states = agent.all_states
        observations = agent.all_observations  # we expect observations to be indexed

        if depth >= self._planning_horizon or self._discount_factor**depth < self._epsilon:  # 終了条件,  探索深さが予め指定した深さを超えたら終了
            return [_PolicyTreeNode(a, depth, agent, self._discount_factor)
                    for a in actions]
        else:
            # 各観測は K 個のサブポリシーツリーにつながる可能性があり、これは _build_policy_trees の出力と一致
            # 次に、一連の観測に対して、ポリシーツリーは、K 個の可能なサブポリシーツリーのプールから
            # 各観測ごとに1つのサブポリシーツリーを組み合わせることにで形成される
            # これらのサブポリシーツリーの集合のデカルト積を取り、個々のポリシーツリーを構築
            groups = [self._build_policy_trees(depth+1, agent)
                      for i in range(len(observations))]
            # (Sanity check) We expect all groups to have same size
            group_size = len(groups[0])
            for g in groups:
                assert group_size == len(g)

            # This computes all combinations of sub policy trees. Each combination
            # will become one policy tree that will be returned, with an action to
            # take at the current depth level as the root.
            combinations = itertools.product(*([torch.arange(group_size)]*len(observations)))  # 返り値はtuple
            policy_trees = []
            for comb in combinations:
                # comb is a tuple of indicies, e.g. (i, j, k) that means
                # for observation 0, the sub policy tree is at index i of its group;
                # for observation 1, the sub policy tree is at index j of its group, etc.
                # We want to create a mapping from observation to sub policy tree.
                assert len(comb) == len(observations)  # sanity check
                children = {}
                for oi in range(len(comb)):
                    sub_policy_tree = groups[oi][comb[oi]]
                    # children[observations[oi]] = sub_policy_tree
                    print("type(observations[oi]):", type(observations[oi]))
                    obs_key = observations[oi]
                    print("type(obs_key):", type(obs_key))
                    children[obs_key] = sub_policy_tree

                # Now that we have the children, we know that there could be
                # |A| different root nodes, resulting in |A| different policy trees
                for a in actions:
                    policy_tree_node = _PolicyTreeNode(
                        a, depth, agent, self._discount_factor, children=children)
                    policy_trees.append(policy_tree_node)

            return policy_trees
