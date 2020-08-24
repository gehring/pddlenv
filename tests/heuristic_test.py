import pddlenv


def test_heuristic(problem):
    preds = {p.__name__: p for p in problem.predicates}
    literals = frozenset({preds["Dirty"]("house"), preds["Out"]("cat", "house")})
    heuristic = pddlenv.Heuristic("hadd")
    assert heuristic(literals, problem) == 1
