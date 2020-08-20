from typing import Collection, Dict, Sequence

import numpy as np

from pddlenv.array.indexing import compute_indices
from pddlenv.base import Literal
from pddlenv.lifted import Problem


def to_dense_binary(literals: Sequence[Collection[Literal]],
                    problem: Problem,
                    dtype: type = np.float32) -> Dict[int, np.ndarray]:
    indices, shapes = compute_indices(
        literals, problem.objectmap.objects, problem.domain.predicates)

    n = len(literals)
    features = {arity: np.zeros((n,) + shape, dtype=dtype) for arity, shape in shapes.items()}
    for k, idx in indices.items():
        features[k][idx] = 1

    return features


def to_flat_dense_binary(literals: Sequence[Collection[Literal]],
                         problem: Problem,
                         dtype: type = np.float32) -> np.ndarray:
    features = to_dense_binary(literals, problem, dtype=dtype)
    return np.concatenate(tuple(x.reshape((x.shape[0], -1)) for x in features.values()), axis=-1)
