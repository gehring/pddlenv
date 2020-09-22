from typing import Sequence, Union

import numpy as np

from pddlenv import env
from pddlenv.generators import base


def as_state_initializer(seed: Union[int, Sequence[int]],
                         problems: base.ProblemSampler,
                         literals: base.LiteralsSampler) -> env.StateInitializer:
    child_seeds = np.random.SeedSequence(seed).spawn(2)
    problem_rng, literals_rng = (np.random.default_rng(s) for s in child_seeds)

    problem_generator = problems.generator(problem_rng)
    problem = None
    while True:
        if problem is None:
            problem = next(problem_generator)
        init_literals = literals.sample(problem, literals_rng)
        problem = yield env.EnvState(init_literals, problem)
