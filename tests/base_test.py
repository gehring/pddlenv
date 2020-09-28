def test_problem(dummy_problem):
    actions = dummy_problem.grounded_actions
    assert len(actions) == 2

    preds = {p.__name__: p for p in dummy_problem.predicates}
    literals = {preds["Clean"]("house"), preds["Out"]("cat", "house")}
    valid_actions = dummy_problem.valid_actions(literals)
    assert len(valid_actions) == 1
