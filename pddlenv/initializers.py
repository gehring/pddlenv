import glob
import os
from typing import AbstractSet, Optional, Sequence

from .base import Predicate, Problem
from .env import EnvState, StateInitializer, reachable_states
from .parsing import parse_pddl_problem


def pddlgym_initializer(rng,
                        domain_filepath: str,
                        problem_dirpath: str,
                        problem_index: Optional[int] = None) -> StateInitializer:
    initial_states = []
    for problem_filepath in sorted(glob.iglob(os.path.join(problem_dirpath, "*.pddl"))):
        initial_states.append(
            EnvState(*parse_pddl_problem(domain_filepath, problem_filepath)))

    while True:
        if problem_index is None:
            init_state = rng.choice(initial_states)
        else:
            init_state = initial_states[problem_index]
        yield init_state


def fixed_problem_initializer(rng,
                              problem: Problem,
                              initial_literals: Sequence[AbstractSet[Predicate]],
                              ) -> StateInitializer:
    while True:
        yield EnvState(rng.choice(initial_literals), problem)


def reachable_states_initializer(rng,
                                 domain_filepath: str,
                                 problem_filepath: str) -> StateInitializer:
    initial_state = EnvState(*parse_pddl_problem(domain_filepath, problem_filepath))
    states = tuple(reachable_states([initial_state]))
    while True:
        yield rng.choice(states)
