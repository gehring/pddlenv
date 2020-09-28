import collections
from typing import Deque, Mapping, Optional, Sequence, Tuple

from pddlenv import env
from pddlenv.base import Action


def generate_plan(end_state: env.EnvState,
                  parents: Mapping[env.EnvState, Optional[Tuple[env.EnvState, Action]]]
                  ) -> Sequence[Action]:
    plan: Deque[Action] = collections.deque()
    state = end_state
    while (stateaction := parents.get(state)) is not None:
        state, action = stateaction
        plan.append(stateaction[1])
    plan.reverse()
    return plan


def generate_path(state: env.EnvState, plan: Sequence[Action]) -> Sequence[env.EnvState]:
    dynamics = env.PDDLDynamics()
    path = [state]
    for action in plan:
        state = dynamics(state, action).observation
        path.append(state)
    return path
