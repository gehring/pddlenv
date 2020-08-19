import collections
import itertools
import operator
from typing import Collection, Dict, Sequence, Tuple

import numpy as np

from pddlenv.base import Literal, PDDLObject, Predicate

IntTup = Tuple[int, ...]
IntTupTup = Tuple[Tuple[int, ...], ...]


def _grounded_literal_index(literal, sorted_objects, sorted_predicates):
    index = tuple(sorted_objects[o] for o in literal.objects)
    return index + (sorted_predicates[literal.predicate],)


def compute_indices(literals: Sequence[Collection[Literal]],
                    objects: Collection[PDDLObject],
                    predicates: Collection[Predicate],
                    ) -> Tuple[Dict[str, IntTupTup], Dict[str, IntTup]]:
    grouped_pred = itertools.groupby(sorted(predicates, key=operator.attrgetter("arity")),
                                     key=operator.attrgetter("arity"))
    sorted_pred = {
        arity: {p: i for i, p in enumerate(sorted(p))}
        for arity, p in grouped_pred
    }

    objects = {o: i for i, o in enumerate(sorted(objects))}

    indices = collections.defaultdict(list)
    for i, lits in enumerate(literals):
        for lit in lits:
            arity = lit.predicate.arity
            indices[arity].append((i,) + _grounded_literal_index(lit, objects, sorted_pred[arity]))

    shapes = {
        str(arity): (len(objects),) * arity + (len(sorted_pred[arity]),)
        for arity in sorted_pred
    }
    tupled_indices = {str(k): tuple(zip(*idx)) for k, idx in indices.items()}

    return tupled_indices, shapes


def ravel_literal_indices(indices: Dict[str, IntTupTup],
                          shapes: Dict[str, IntTup]) -> Tuple[np.ndarray, np.ndarray]:

    arity_offsets = dict(zip(
        shapes.keys(),
        np.cumsum([0] + [np.prod(shape) for shape in list(shapes.values())[:-1]])
    ))
    batch_idx, flat_idx = zip(*(
        (idx[0], np.ravel_multi_index(idx[1:], shapes[arity]) + arity_offsets[arity])
        for arity, idx in indices.items()
    ))
    return np.concatenate(batch_idx), np.concatenate(flat_idx)
