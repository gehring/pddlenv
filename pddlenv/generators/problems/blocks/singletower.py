import dataclasses
import os
from typing import Dict, Iterator, Optional

import numpy as np

from pddlenv import parsing
from pddlenv.base import Problem
from pddlenv.generators import base


@dataclasses.dataclass
class SingleTowerGenerator(base.ProblemSampler):
    min_blocks: int
    max_blocks: int
    round_robin: bool
    blocks_domain_path: str = "blocks.pddl"

    def _generate(self, rng: Optional[np.random.Generator]) -> Iterator[Problem]:
        round_robin = rng is None
        if not round_robin:
            rng = np.random.default_rng(rng)

        pddl_root = os.environ.get("PDDL_ROOT_DIR", "~/pddl")
        blocks_domain_path = os.path.expanduser(os.path.join(pddl_root, self.blocks_domain_path))
        Blocks = parsing.parse_pddl_domain(blocks_domain_path)

        types = {t.__name__: t for t in Blocks.types}
        preds = {p.__name__: p for p in Blocks.predicates}

        problems: Dict[int, Problem] = {}
        num_blocks = self.min_blocks
        blocks_span = self.max_blocks - self.min_blocks + 1
        while True:
            if self.min_blocks == self.max_blocks:
                num_blocks = self.min_blocks
            elif rng is not None:
                num_blocks = rng.integers(self.min_blocks, self.max_blocks, endpoint=True)

            problem = problems.get(num_blocks, None)

            if problem is None:
                blocks = [types["block"](str(i)) for i in range(num_blocks)]
                goal = [preds["on"](x, y) for x, y in zip(blocks[:-1], blocks[1:])]
                problem = Blocks(blocks, goal, {})
                problems[num_blocks] = problem

            yield problem

            if round_robin:
                num_blocks = (num_blocks - self.min_blocks + 1) % blocks_span
                num_blocks += self.min_blocks

    def generator(self, rng: Optional[np.random.Generator] = None) -> Iterator[Problem]:
        if self.round_robin:
            rng = None
        yield from self._generate(rng)

    def enumerate_problems(self) -> Iterator[Problem]:
        yield from self._generate(rng=None)
