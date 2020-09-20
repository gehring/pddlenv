import collections
import itertools
import operator
from typing import Collection, Dict, Iterable, Tuple, Type

import numpy as np

from pddlenv.base import PDDLObject, Predicate

IntTup = Tuple[int, ...]
IntTupTup = Tuple[Tuple[int, ...], ...]


def _grounded_literal_index(literal, sorted_objects, sorted_predicates):
    index = tuple(sorted_objects[o] for o in literal.objects)
    return index + (sorted_predicates[type(literal)],)


def _shape_from_grouped_predicates(num_objects: int,
                                   grouped_pred: Iterable[Tuple[int, Collection[Type[Predicate]]]]):
    return {
        arity: (num_objects,) * arity + (len(preds),)
        for arity, preds in grouped_pred
    }


def compute_shapes(num_objects: int,
                   predicates: Collection[Type[Predicate]]) -> Dict[int, Tuple[int, ...]]:
    grouped_pred = itertools.groupby(sorted(predicates, key=operator.attrgetter("arity")),
                                     key=operator.attrgetter("arity"))
    return _shape_from_grouped_predicates(
        num_objects,
        grouped_pred=((arity, tuple(preds)) for arity, preds in grouped_pred),
    )


def compute_indices(literals: Iterable[Collection[Predicate]],
                    objects: Collection[PDDLObject],
                    predicates: Collection[Type[Predicate]],
                    ) -> Tuple[Dict[int, IntTupTup], Dict[int, IntTup]]:
    grouped_pred = itertools.groupby(sorted(predicates, key=operator.attrgetter("arity")),
                                     key=operator.attrgetter("arity"))
    sorted_pred = {
        arity: {p: i for i, p in enumerate(sorted(p, key=operator.attrgetter("__name__")))}
        for arity, p in grouped_pred
    }

    objects = {o: i for i, o in enumerate(sorted(objects))}

    indices = collections.defaultdict(list)
    for i, lits in enumerate(literals):
        for lit in lits:
            arity = lit.arity
            indices[arity].append((i,) + _grounded_literal_index(lit, objects, sorted_pred[arity]))

    shapes = _shape_from_grouped_predicates(len(objects), sorted_pred.items())
    tupled_indices = {k: tuple(zip(*idx)) for k, idx in indices.items()}

    return tupled_indices, shapes


def ravel_literal_indices(indices: Dict[int, IntTupTup],
                          shapes: Dict[int, IntTup]) -> Tuple[np.ndarray, np.ndarray]:

    arity_offsets = dict(zip(
        shapes.keys(),
        np.cumsum([0] + [np.prod(shape) for shape in list(shapes.values())[:-1]])
    ))
    batch_idx, flat_idx = zip(*(
        (idx[0], np.ravel_multi_index(idx[1:], shapes[arity]) + arity_offsets[arity])
        for arity, idx in indices.items()
    ))
    return np.concatenate(batch_idx), np.concatenate(flat_idx)
