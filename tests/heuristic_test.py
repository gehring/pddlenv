import pytest

import pddlenv


def test_dummy_heuristic(dummy_problem):
    preds = {p.__name__: p for p in dummy_problem.predicates}
    literals = frozenset({preds["Dirty"]("house"), preds["Out"]("cat", "house")})
    heuristic = pddlenv.Heuristic("hadd")
    assert heuristic(literals, dummy_problem) == 1


@pytest.mark.parametrize("heuristic_name", ["hadd", "hff"])
def test_heuristic(heuristic_name, pddl_test_case):
    heuristic = pddlenv.Heuristic(heuristic_name)
    heuristic(pddl_test_case.init_literals, pddl_test_case.problem)
