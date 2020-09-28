import dataclasses
from typing import Any, Mapping, Protocol

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


class Logger(Protocol):

    def write(self, data: Mapping[str, Any]):
        raise NotImplementedError
