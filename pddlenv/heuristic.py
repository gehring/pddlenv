import dataclasses
import functools
import itertools
from typing import AbstractSet, Callable, FrozenSet, Optional, Tuple

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

    def _heuristic(literals: FrozenSet[Predicate]) -> float:
        node = Node(literals & task.facts)
        return heuristic(node)

    if cache_maxsize != 0:
        _heuristic = functools.lru_cache(maxsize=cache_maxsize)(_heuristic)

    return _heuristic


@dataclasses.dataclass(frozen=True)
class Heuristic:
    name: str
    discount: float = dataclasses.field(default=1.)
    num_cached_problems: Optional[int] = dataclasses.field(default=None)
    cache_size_per_problem: Optional[int] = dataclasses.field(default=None)
    _heuristic_function: Callable = dataclasses.field(init=False)

    def __post_init__(self):
        # wrap make_heuristic_function with the name and size arguments set
        make_heuristic = functools.partial(
            make_heuristic_function,
            self.name,
            cache_maxsize=self.cache_size_per_problem,
        )
        # wrap make_heuristic_function with a lru cache if size is None or non-zero
        if self.num_cached_problems != 0:
            make_heuristic = functools.lru_cache(self.num_cached_problems)(make_heuristic)

        object.__setattr__(self, "_heuristic_function", make_heuristic)

    def __call__(self, literals: AbstractSet[Predicate], problem: Problem) -> float:
        h_val = self._heuristic_function(problem)(literals)
        if self.discount < 1:
            h_val = (1 - self.discount**h_val)/(1 - self.discount)
        return h_val
