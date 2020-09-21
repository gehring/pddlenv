import operator

import numpy as np

import pddlenv


def test_compute_index(problem):
    indices, shapes1 = pddlenv.array.compute_indices([], problem.objects, problem.predicates)
    assert len(indices) == 0

    indices, shapes2 = pddlenv.array.compute_indices([[]], problem.objects, problem.predicates)
    assert len(indices) == 0
    assert shapes1 == shapes2

    types = {t.__name__: t for t in problem.types}
    preds = {p.__name__: p for p in problem.predicates}
    literals = [
        [preds["Clean"](types["House"]("house"))],
        [preds["Clean"](types["House"]("house")),
         preds["In"](types["Cat"]("cat"), types["Hat"]("hat"))],
    ]
    indices, shapes3 = pddlenv.array.compute_indices(literals, problem.objects, problem.predicates)
    assert shapes2 == shapes3

    # check that the `Clean` predicate shared between both gives the same indices
    assert operator.eq(*indices[1][1])
    assert operator.eq(*indices[1][2])

    # check that the correct objects are indexed
    assert problem.objects[indices[1][1][0]] == "house"
    assert problem.objects[indices[2][1][0]] == "cat"
    assert problem.objects[indices[2][2][0]] == "hat"

    # Check that the correct predicates are indexed. We expect the predicates to already be sorted.
    unary_preds = [p for p in problem.predicates if p.arity == 1]
    binary_preds = [p for p in problem.predicates if p.arity == 2]
    assert unary_preds[indices[1][2][0]] == preds["Clean"]
    assert binary_preds[indices[2][3][0]] == preds["In"]


def test_compute_action_index(problem):
    types = {t.__name__: t for t in problem.types}
    actions = {a.__name__: a for a in problem.actions}
    literals = [
        [actions["CleanHouse"](types["Cat"]("cat"), types["House"]("house"))],
        [actions["CleanHouse"](types["Cat"]("cat"), types["House"]("house")),
         actions["LetIn"](types["Cat"]("cat"), types["Hat"]("hat"), types["House"]("house"))],
    ]
    indices, shapes = pddlenv.array.compute_indices(literals, problem.objects, problem.actions)
    assert shapes == {2: (3, 3, 1), 3: (3, 3, 3, 1)}

    # check that the `CleanHouse` predicate shared between both gives the same indices
    assert operator.eq(*indices[2][1])
    assert operator.eq(*indices[2][2])
    assert operator.eq(*indices[2][3])

    # check that the correct objects are indexed
    assert problem.objects[indices[2][1][0]] == "cat"
    assert problem.objects[indices[2][2][0]] == "house"
    assert problem.objects[indices[3][1][0]] == "cat"
    assert problem.objects[indices[3][2][0]] == "hat"
    assert problem.objects[indices[3][3][0]] == "house"

    # Check that the correct actions are indexed. We expect the predicates to already be sorted.
    binary_preds = [p for p in problem.actions if p.arity == 2]
    trenary_preds = [p for p in problem.actions if p.arity == 3]
    assert binary_preds[indices[2][-1][0]] == actions["CleanHouse"]
    assert trenary_preds[indices[3][-1][0]] == actions["LetIn"]


def test_ravel_unravel_idempotent(problem):
    types = {t.__name__: t for t in problem.types}
    preds = {p.__name__: p for p in problem.predicates}
    literals = [
        [preds["Clean"](types["House"]("house"))],
        [preds["Clean"](types["House"]("house")),
         preds["In"](types["Cat"]("cat"), types["Hat"]("hat"))],
    ]
    indices, shapes = pddlenv.array.compute_indices(literals, problem.objects, problem.predicates)

    flat_idx = pddlenv.array.ravel_literal_indices(indices, shapes)
    unravelled = pddlenv.array.unravel_literal_indices(flat_idx, shapes)

    # check that indices are unravelled to the same value and in the same order
    for (arity, unravelled_idx), idx in zip(unravelled.items(), indices.values()):
        np.testing.assert_array_equal(
            unravelled_idx,
            idx,
            f"Unravelled indices not equal to original indices for arity {arity}",
        )

    # check that unravelled indices ravel to the same values
    np.testing.assert_array_equal(pddlenv.array.ravel_literal_indices(unravelled, shapes), flat_idx)
