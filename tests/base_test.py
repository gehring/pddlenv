def test_problem(problem):
    actions = problem.grounded_actions
    assert len(actions) == 2

    preds = {p.__name__: p for p in problem.predicates}
    literals = {preds["Clean"]("house"), preds["Out"]("cat", "house")}
    valid_actions = problem.valid_actions(literals)
    assert len(valid_actions) == 1
