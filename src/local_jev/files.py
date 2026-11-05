"""Reading and writing the JSON files of bench runs, reports and calibrations."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from itertools import count
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    """One value per non-blank line."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def utc_stamp() -> str:
    """The time now, as a file name starts with it: `20261105T103000Z`."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def create_new[T](create: Callable[[Path], T], directory: Path, name: str, suffix: str = "") -> T:
    """`create` called on the first of `<directory>/<name><suffix>`, `<name>-2<suffix>`, … that
    does not exist yet; `create` makes it, raising FileExistsError if another got there first.

    So nothing is ever written over.
    """
    directory.mkdir(parents=True, exist_ok=True)
    for attempt in count(1):
        path = directory / (f"{name}{suffix}" if attempt == 1 else f"{name}-{attempt}{suffix}")
        try:
            return create(path)
        except FileExistsError:
            continue
