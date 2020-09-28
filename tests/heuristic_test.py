import pddlenv


def test_heuristic(dummy_problem):
    preds = {p.__name__: p for p in dummy_problem.predicates}
    literals = frozenset({preds["Dirty"]("house"), preds["Out"]("cat", "house")})
    heuristic = pddlenv.Heuristic("hadd")
    assert heuristic(literals, dummy_problem) == 1
