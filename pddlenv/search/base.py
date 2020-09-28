import dataclasses

from pddlenv import base, env


@dataclasses.dataclass(order=True)
class Candidate:
    heuristic: float
    state: env.EnvState = dataclasses.field(compare=False)


@dataclasses.dataclass(order=True)
class SuccessorCandidate:
    heuristic: float
    state: env.EnvState = dataclasses.field(compare=False)
    action: base.Action = dataclasses.field(compare=False)
