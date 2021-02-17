import dataclasses
import functools
import itertools
from typing import AbstractSet, FrozenSet, Optional, Tuple

import pyperplan

from .base import Action, Predicate, Problem


@dataclasses.dataclass(frozen=True)
class Task:
    facts: FrozenSet[Predicate]
    goals: FrozenSet[Predicate]
    operators: Tuple[Action, ...]
    initial_state: Optional[FrozenSet[Predicate]] = None


@dataclasses.dataclass(frozen=True)
class Node:
    state: FrozenSet[Predicate]


def make_heuristic_function(name: str, problem: Problem, cache_maxsize: Optional[int] = None):
    actions = problem.grounded_actions
    task = Task(
        facts=frozenset(itertools.chain(*(a.literals() for a in actions))),
        goals=problem.goal,
        operators=actions,
    )
    heuristic = pyperplan.HEURISTICS[name](task)

    @functools.lru_cache(maxsize=cache_maxsize)
    def _heuristic(literals: FrozenSet[Predicate]) -> float:
        node = Node(literals & task.facts)
        return heuristic(node)

    return _heuristic


@dataclasses.dataclass(frozen=True)
class Heuristic:
    name: str
    discount: float = dataclasses.field(default=1.)

    def __call__(self, literals: AbstractSet[Predicate], problem: Problem) -> float:
        h_val = self._heuristic_function(problem)(literals)
        if self.discount < 1:
            h_val = (1 - self.discount**h_val)/(1 - self.discount)
        return h_val

    @functools.lru_cache
    def _heuristic_function(self, problem, cache_maxsize=None):
        return make_heuristic_function(self.name, problem, cache_maxsize=cache_maxsize)
