from typing import Collection, Sequence

import numpy as np

from pddlenv.array.indexing import compute_indices
from pddlenv.base import Literal
from pddlenv.lifted import Problem


def dense_binary_encoding(literals: Sequence[Collection[Literal]],
                          problem: Problem,
                          dtype: type = np.float32) -> np.ndarray:
    indices, shapes = compute_indices(
        literals, problem.objectmap.objects, problem.domain.predicates)

    features_dtype = [(k, dtype, shape) for k, shape in shapes.items()]
    features = np.zeros(len(literals), dtype=features_dtype)
    for k, idx in indices.items():
        features[k][idx] = 1

    return features
