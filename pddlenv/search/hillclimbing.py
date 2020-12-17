import heapq
import time
from typing import AbstractSet, Callable, Dict, Optional, Sequence, Tuple

from pddlenv import env
from pddlenv.base import Action, Predicate, Problem
from pddlenv.search import base, utils

Heuristic = Callable[[AbstractSet[Predicate], Problem], float]


class HillClimbing:

    def __init__(self, heuristic: Heuristic, logger: Optional[base.Logger] = None):
        self.heuristic = heuristic
        self.logger = logger

    def search(self,
               state: env.EnvState,
               time_limit: float = None,
               expansion_limit: int = None) -> Optional[Sequence[Action]]:
        expanded_states = 0
        start_time = time.perf_counter()

        # we assume that the dynamics will never change the problem instance
        problem = state.problem
        heap = [base.Candidate(0., state)]
        parents: Dict[env.EnvState, Optional[Tuple[env.EnvState, Action]]] = {state: None}
        dynamics = env.PDDLDynamics()

        while heap:
            if time_limit and time_limit <= time.perf_counter() - start_time:
                break
            if expansion_limit and expansion_limit <= expanded_states:
                break

            expanded_states += 1
            state = heapq.heappop(heap).state
            heap = []           # hill climbing --- forget everything
            actions, timesteps = dynamics.sample_transitions(state)
            next_states = [timestep.observation for timestep in timesteps]

            for action, next_state in zip(actions, next_states):
                literals = next_state.literals

                if next_state not in parents:
                    parents[next_state] = (state, action)
                    heapq.heappush(
                        heap, base.Candidate(self.heuristic(literals, problem), next_state))

                if problem.goal_satisfied(literals):
                    if self.logger is not None:
                        self.logger.write({"expanded_states": expanded_states,
                                           "search_time": time.perf_counter() - start_time})
                    return utils.generate_plan(next_state, parents)

        if self.logger is not None:
            self.logger.write({"expanded_states": expanded_states,
                               "search_time": time.perf_counter() - start_time})
        return None
