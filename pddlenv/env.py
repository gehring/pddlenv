import dataclasses
from typing import AbstractSet, Generator, Optional, Tuple

import dm_env

from .base import Action, Literal
from .heuristic import Heuristic
from .lifted import Problem

StateInitializer = Generator["EnvState", Optional[Problem], None]


class InvalidAction(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class EnvState:
    __slots__ = ("literals", "problem")
    literals: AbstractSet[Literal]
    problem: Problem


@dataclasses.dataclass
class PDDLDynamics:
    discount: float
    use_cost_reward: bool
    heuristic: Optional[Heuristic] = None

    def __call__(self, state: EnvState, action: Action) -> dm_env.TimeStep:
        literals = state.literals
        problem = state.problem
        if not action.applicable(literals):
            raise InvalidAction(
                f"Preconditions not satisfied.\n\nAction: {action}\n\nState literals: {literals}")

        next_literals = action.apply(literals)
        goal_reached = problem.goal_satisfied(next_literals)

        reward = -1 if self.use_cost_reward else 0.
        if goal_reached:
            reward += 1

        if self.heuristic is not None:
            shaping_reward = self.heuristic(literals, problem)
            if not goal_reached:
                shaping_reward -= self.discount * self.heuristic(next_literals, problem)
            reward += shaping_reward

        next_state = dataclasses.replace(state, literals=next_literals)
        if goal_reached:
            timestep = dm_env.termination(reward, next_state)
        else:
            timestep = dm_env.transition(reward, next_state, self.discount)
        return timestep

    def sample_transitions(self, state: EnvState) -> Tuple[dm_env.TimeStep, ...]:
        actions = state.problem.valid_actions(state.literals)
        return tuple(self(state, a) for a in actions)


class PDDLEnv(dm_env.Environment):
    dynamics: PDDLDynamics
    state_initializer: StateInitializer
    state: Optional[EnvState] = None

    def reset(self):
        pass

    def step(self, action):
        pass

    def observation_spec(self):
        pass

    def action_spec(self):
        pass
