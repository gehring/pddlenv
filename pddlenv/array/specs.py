import dataclasses
import itertools
import operator
from typing import Collection, Dict, Optional, Tuple, Union

from pddlenv.base import Predicate
from pddlenv.lifted import Problem

NUM_OBJECTS = "# objects"


@dataclasses.dataclass(frozen=True)
class LiteralArray(object):
    __slots__ = ("shape", "dtype", "name")
    shape: Dict[str, Tuple[Union[str, int], ...]]
    dtype: type
    name: Optional[str]

    def __init__(self, predicates: Collection[Predicate], dtype, name=None):
        grouped_pred = itertools.groupby(sorted(predicates, key=operator.attrgetter("arity")),
                                         key=operator.attrgetter("arity"))
        super().__setattr__("shape", {
            str(arity): (NUM_OBJECTS,) * arity + (len(tuple(preds)),)
            for arity, preds in grouped_pred
        })
        super().__setattr__("dtype", dtype)
        super().__setattr__("name", name)

    def grounded_shape(self, problem: Problem):
        num_objects = len(problem.objects)
        return {
            arity: (num_objects,) * int(arity) + shape[-1:]
            for arity, shape in self.shape.items()
        }
