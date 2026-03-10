from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Level:
    level: int
    label: str
    description: str


@dataclass
class Criterion:
    id: str
    name: str
    levels: List[Level]

    def get_level(self, level: int) -> Level | None:
        for l in self.levels:
            if l.level == level:
                return l
        return None


@dataclass
class Rubric:
    criteria: List[Criterion]

    def get_criterion(self, criterion_id: str) -> Criterion | None:
        for c in self.criteria:
            if c.id == criterion_id:
                return c
        return None

