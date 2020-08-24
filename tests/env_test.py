import pytest

import pddlenv


def test_dynamics(problem):
    actions = {a.__name__: a for a in problem.actions}
    preds = {p.__name__: p for p in problem.predicates}
    literals = {preds["Dirty"]("house"), preds["Out"]("cat", "house")}
    dynamics = pddlenv.PDDLDynamics()

    state = pddlenv.EnvState(literals, problem)
    timestep = dynamics(state, actions["CleanHouse"]("cat", "house", problem=problem))
    new_state = timestep.observation
    assert "(Clean house)" in new_state.literals
    assert "(Out cat house)" in new_state.literals
    assert "(Dirty house)" not in new_state.literals

    with pytest.raises(pddlenv.InvalidAction):
        dynamics(new_state, actions["CleanHouse"]("cat", "house", problem=problem))
