import itertools
from typing import Collection, Sequence, Tuple

import numpy as np

from pddlenv.base import Literal, PDDLObject, Predicate

IntTupTup = Tuple[Tuple[int, ...], ...]


def _grounded_literal_index(literal, sorted_objects, sorted_predicates):
    index = tuple(sorted_objects[o] for o in literal.objects)
    return index + (sorted_predicates[literal.predicate],)


def compute_indices(literals: Sequence[Collection[Literal]],
                    objects: Collection[PDDLObject],
                    predicates: Collection[Predicate]) -> Tuple[Sequence[IntTupTup], IntTupTup]:
    arity_key = lambda p: p.arity
    grouped_pred = itertools.groupby(sorted(predicates, key=arity_key),
                                     key=arity_key)
    sorted_pred = {
        arity: {p: i for i, p in enumerate(sorted(p))}
        for arity, p in grouped_pred
    }

    objects = {o: i for i, o in enumerate(sorted(objects))}
    arity_indices = {
        arity: i
        for i, arity in enumerate(sorted(sorted_pred.keys()))
    }

    def _compute_index(lit):
        arity = lit.predicate.arity
        lit_index = _grounded_literal_index(lit, objects, sorted_pred[arity])
        return (arity_indices[arity],) + lit_index

    indices = [tuple(_compute_index(lit) for lit in lits) for lits in literals]
    shapes = tuple((len(objects),) * arity + (len(sorted_pred[arity]),) for arity in arity_indices)

    return indices, shapes


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
