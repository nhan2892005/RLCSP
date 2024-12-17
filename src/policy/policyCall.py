from .policy import Policy
from .impl.best_fit import BestFit
from .impl.generate_column import GenerateColumn
from .impl.genetic import GeneticAlgorithm
from .impl.treebased import TreeBasedHeuristic
from .impl.random import RandomPolicy
from .impl.greedy import GreedyPolicy

arr_policy = [BestFit, GenerateColumn, GeneticAlgorithm, TreeBasedHeuristic, RandomPolicy, GreedyPolicy]

class PolicyCall(Policy):
    def __init__(self, policy_id=1):
        assert policy_id in range(len(arr_policy)), "Invalid policy id"

        self.policy = arr_policy[policy_id]()

    def get_action(self, observation, info):
        return self.policy.get_action(observation, info)