from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List

from .models import Criterion, Level, Rubric


def _load_rubric_dict(path: Path | None = None) -> Dict[str, Any]:
    """
    Load rubric JSON as dictionary.

    If `path` is None, loads the default `rubric.json` shipped with the package.
    """
    if path is None:
        with resources.files(__package__).joinpath("rubric.json").open("r", encoding="utf-8") as f:
            return json.load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_rubric(path: str | Path | None = None) -> Rubric:
    """
    Load rubric JSON and return a `Rubric` object.
    """
    rubric_dict = _load_rubric_dict(Path(path) if path is not None else None)
    criteria: List[Criterion] = []
    for c in rubric_dict.get("criteria", []):
        levels = [
            Level(
                level=int(l["level"]),
                label=str(l["label"]),
                description=str(l["description"]),
            )
            for l in c.get("levels", [])
        ]
        criteria.append(
            Criterion(
                id=str(c["id"]),
                name=str(c["name"]),
                levels=levels,
            )
        )
    return Rubric(criteria=criteria)

