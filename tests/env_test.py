import pytest

import pddlenv


def test_dynamics(dummy_problem):
    actions = {a.__name__: a for a in dummy_problem.actions}
    preds = {p.__name__: p for p in dummy_problem.predicates}
    literals = {preds["Dirty"]("house"), preds["Out"]("cat", "house")}
    dynamics = pddlenv.PDDLDynamics()

    state = pddlenv.EnvState(literals, dummy_problem)
    timestep = dynamics(state, actions["CleanHouse"]("cat", "house", problem=dummy_problem))
    new_state = timestep.observation
    assert "(Clean house)" in new_state.literals
    assert "(Out cat house)" in new_state.literals
    assert "(Dirty house)" not in new_state.literals

    with pytest.raises(pddlenv.InvalidAction):
        dynamics(new_state, actions["CleanHouse"]("cat", "house", problem=dummy_problem))
