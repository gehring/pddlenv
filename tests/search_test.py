import pytest

import pddlenv
from pddlenv.search import utils


def test_greedy_best_first(pddl_test_case):
    if not pddl_test_case.has_solution():
        pytest.skip(f"Test case has no ground truth to compare to. {pddl_test_case}")

    problem = pddl_test_case.problem
    init_state = pddlenv.EnvState(pddl_test_case.init_literals, problem)
    bfs = pddlenv.search.GreedyBestFirst(pddlenv.Heuristic("hadd"))
    plan = bfs.search(init_state)
    assert (pddl_test_case.path_length > 0) == (plan is not None)

    if plan is not None:
        path = utils.generate_path(init_state, plan)
        assert path[0] == init_state
        assert problem.goal_satisfied(path[-1].literals)
