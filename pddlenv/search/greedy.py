import heapq
from typing import AbstractSet, Callable, Dict, Optional, Sequence, Tuple

from pddlenv import env
from pddlenv.base import Action, Predicate, Problem
from pddlenv.search import base, utils

Heuristic = Callable[[AbstractSet[Predicate], Problem], float]


class GreedyBestFirst:

    def __init__(self, heuristic: Heuristic):
        self.heuristic = heuristic

    def search(self, state: env.EnvState) -> Optional[Sequence[Action]]:
        # we assume that the dynamics will never change the problem instance
        problem = state.problem
        heap = [base.Candidate(0., state)]
        parents: Dict[env.EnvState, Optional[Tuple[env.EnvState, Action]]] = {state: None}
        dynamics = env.PDDLDynamics()

        while heap:
            state = heapq.heappop(heap).state
            actions, timesteps = dynamics.sample_transitions(state)
            next_states = [timestep.observation for timestep in timesteps]
            for action, next_state in zip(actions, next_states):
                literals = next_state.literals
                if next_state not in parents:
                    parents[next_state] = (state, action)
                    heapq.heappush(
                        heap, base.Candidate(self.heuristic(literals, problem), next_state))
                if problem.goal_satisfied(literals):
                    return utils.generate_plan(next_state, parents)
        return None
