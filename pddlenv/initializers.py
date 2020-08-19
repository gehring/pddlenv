import glob
import os
from typing import AbstractSet, Optional, Sequence

import pyperplan

from .base import Literal
from .env import EnvState, StateInitializer
from .lifted import Problem, parse_pyperplan_literals


def pddlgym_initializer(rng,
                        domain_filepath: str,
                        problem_dirpath: str) -> StateInitializer:
    parser = pyperplan.Parser(domain_filepath)
    pyperplan_domain = parser.parse_domain()

    initial_envstates = []
    for problem_filepath in glob.iglob(os.path.join(problem_dirpath, "*.pddl")):
        parser.probFile = problem_filepath
        pyperplan_problem = parser.parse_problem(pyperplan_domain)

        problem = Problem.from_pyperplan(pyperplan_problem)
        initial_literals = parse_pyperplan_literals(pyperplan_problem.initial_state, problem.domain)
        initial_envstates.append(EnvState(initial_literals, problem))

    while True:
        yield rng.choice(initial_envstates)


def fixed_problem_initializer(rng,
                              domain_filepath: str,
                              problem_filepath: str,
                              initial_literals: Optional[Sequence[AbstractSet[Literal]]] = None,
                              ) -> StateInitializer:
    parser = pyperplan.Parser(domain_filepath, problem_filepath)
    pyperplan_domain = parser.parse_domain()
    pyperplan_problem = parser.parse_problem(pyperplan_domain)

    problem = Problem.from_pyperplan(pyperplan_problem)
    if initial_literals is None:
        initial_literals = [
            parse_pyperplan_literals(pyperplan_problem.initial_state, problem.domain),
        ]

    while True:
        if len(initial_literals) > 1:
            initial_envstate = EnvState(rng.choice(initial_literals), problem)
        else:
            initial_envstate = EnvState(initial_literals[0], problem)
        yield initial_envstate
