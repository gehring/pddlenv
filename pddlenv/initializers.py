import glob
import os
from typing import AbstractSet, Sequence

from .base import Literal
from .env import EnvState, StateInitializer, reachable_states
from .lifted import Problem, parse_pddl_problem


def pddlgym_initializer(rng,
                        domain_filepath: str,
                        problem_dirpath: str) -> StateInitializer:
    initial_states = []
    for problem_filepath in glob.iglob(os.path.join(problem_dirpath, "*.pddl")):
        initial_states.append(
            EnvState(*parse_pddl_problem(domain_filepath, problem_filepath)))

    while True:
        yield rng.choice(initial_states)


def fixed_problem_initializer(rng,
                              problem: Problem,
                              initial_literals: Sequence[AbstractSet[Literal]],
                              ) -> StateInitializer:
    while True:
        yield EnvState(rng.choice(initial_literals), problem)


def reachable_states_initializer(rng,
                                 domain_filepath: str,
                                 problem_filepath: str) -> StateInitializer:
    initial_state = EnvState(*parse_pddl_problem(domain_filepath, problem_filepath))
    states = reachable_states([initial_state])
    while True:
        yield rng.choice(states)
