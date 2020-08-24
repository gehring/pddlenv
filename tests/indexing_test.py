import operator

import pddlenv


def test_compute_indexing(problem):
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
