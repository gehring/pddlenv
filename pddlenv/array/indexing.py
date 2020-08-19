import collections
import itertools
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
                    predicates: Collection[Predicate]
                    ) -> Tuple[Dict[str, IntTupTup], Dict[str, IntTup]]:
    arity_key = lambda p: p.arity
    grouped_pred = itertools.groupby(sorted(predicates, key=arity_key),
                                     key=arity_key)
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


def _ravel_literal_index(arity_offsets: np.ndarray,
                         indices: IntTupTup,
                         shapes: IntTupTup) -> Tuple[np.ndarray, ...]:
    return tuple(
        np.ravel_multi_index(index[1:], shapes[index[0]]) + arity_offsets[index[0]]
        for index in indices
    )


def ravel_literal_indices(indices: Sequence[IntTupTup], shapes: IntTupTup) -> np.ndarray:
    arity_offsets = np.cumsum([0] + [np.prod(shape) for shape in shapes[:-1]])
    return np.stack(tuple(_ravel_literal_index(arity_offsets, idx, shapes) for idx in indices))
