from typing import FrozenSet, Iterator, Optional, Protocol, TypeVar

import numpy as np

import pddlenv

L = TypeVar("L", bound=pddlenv.Predicate, covariant=True)
P = TypeVar("P", bound=pddlenv.Problem, covariant=True)


class ProblemSampler(Protocol[P]):

    def generator(self, rng: Optional[np.random.Generator] = None) -> Iterator[P]:
        raise NotImplementedError

    def enumerate_problems(self) -> Iterator[P]:
        raise NotImplementedError


class LiteralsSampler(Protocol[L]):

    def sample(self,
               problem: pddlenv.Problem,
               rng: Optional[np.random.Generator] = None) -> FrozenSet[L]:
        raise NotImplementedError
