from typing import Collection, Sequence

import numpy as np
import numpy.lib.recfunctions

from pddlenv.array.indexing import compute_indices
from pddlenv.base import Literal
from pddlenv.lifted import Problem


def to_dense_binary(literals: Sequence[Collection[Literal]],
                    problem: Problem,
                    dtype: type = np.float32) -> np.ndarray:
    indices, shapes = compute_indices(
        literals, problem.objectmap.objects, problem.domain.predicates)

    features_dtype = [(k, dtype, shape) for k, shape in shapes.items()]
    features = np.zeros(len(literals), dtype=features_dtype)
    for k, idx in indices.items():
        features[k][idx] = 1

    return features


def to_flat_dense_binary(literals: Sequence[Collection[Literal]],
                         problem: Problem,
                         dtype: type = np.float32) -> np.ndarray:
    features = to_dense_binary(literals, problem, dtype=dtype)
    return np.lib.recfunctions.structured_to_unstructured(features)
