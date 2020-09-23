from typing import FrozenSet, Optional

import numpy as np

from pddlenv.base import Predicate, Problem
from pddlenv.generators import base

BLOCK_TYPE_NAME = "block"
PREDICATE_REQ = {"on", "clear", "handempty"}


class ClearSampler(base.LiteralsSampler):

    def sample(self,
               problem: Problem,
               rng: Optional[np.random.Generator] = None) -> FrozenSet[Predicate]:
        types = [t.__name__ for t in problem.types]
        preds = {p.__name__: p for p in problem.predicates}

        if BLOCK_TYPE_NAME not in types:
            raise ValueError(
                "Only blocks world problems are supported which must contain the following "
                f"types: {BLOCK_TYPE_NAME}\n"
                f"Found types: {set(types)}"
            )
        if not (PREDICATE_REQ - preds.keys()):
            raise ValueError(
                "Only blocks world problems are supported which must contain the following "
                f"predicates: {PREDICATE_REQ}\n"
                f"Found predicates: {set(preds.keys())}"
            )

        blocks = [x for x in problem.objects if type(x).__name__ == BLOCK_TYPE_NAME]
        return frozenset([preds["clear"](x) for x in blocks]
                         + [preds["ontable"](x) for x in blocks]
                         + [preds["handempty"]()])
