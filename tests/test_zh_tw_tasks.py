"""The Traditional-Chinese Task set against jevbench's 231 public Tasks at the pinned commit.

Only the State may differ: ids, Questions, Labels, expected Labels and every other field stay as
jevbench has them, so a bad edit cannot silently change the answer key. jevbench's Tasks are read
from the bench runner's cache (`.cache/`, which `tests/test_bench.py` and every `local-jev bench`
fill); with no cache, it is fetched once. When it cannot be had (offline, no cache), the checks
that need it fail: an integrity check that cannot see its source has not passed. The checks on the
translated States themselves need no jevbench at all.
"""

import json
import re
from pathlib import Path

import pytest

from local_jev.bench import JEVBENCH_COMMIT, TASK_SETS, BenchFailed, fetch_jevbench

PROJECT_CACHE = Path(__file__).parents[1] / ".cache"

PARTS = {"zh-tw-easy": "easy", "zh-tw-original": "original", "zh-tw-hard": "hard"}
"""Each part of the Traditional-Chinese set, and the public part it translates."""


def _load(files: list[tuple[str, Path]]) -> dict[str, list[dict]]:
    return {
        part: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for part, path in files
    }


@pytest.fixture(scope="module")
def zh_tw() -> dict[str, list[dict]]:
    """The set's Tasks by part, as `local-jev bench --task-set zh-tw` reads them."""
    # Committed with local-jev, so they are found without any jevbench checkout.
    return _load(TASK_SETS["zh-tw"](Path("no-jevbench-checkout")))


@pytest.fixture(scope="module")
def public() -> dict[str, list[dict]]:
    """jevbench's public Tasks at the pinned commit, by the Traditional-Chinese part of each."""
    try:
        jevbench = fetch_jevbench(PROJECT_CACHE)
    except BenchFailed as e:
        pytest.fail(f"cannot check the set against jevbench {JEVBENCH_COMMIT[:7]}: {e}")
    by_part = _load(TASK_SETS["public"](jevbench))
    return {zh: by_part[en] for zh, en in PARTS.items()}


def test_the_set_holds_all_231_public_tasks_in_three_parts(zh_tw):
    assert {part: len(tasks) for part, tasks in zh_tw.items()} == {
        "zh-tw-easy": 48,
        "zh-tw-original": 72,
        "zh-tw-hard": 111,
    }


def test_each_part_has_the_same_task_ids_in_the_same_order_as_jevbench(zh_tw, public):
    for part in PARTS:
        assert [t["id"] for t in zh_tw[part]] == [t["id"] for t in public[part]], part


def test_every_task_matches_jevbench_in_everything_but_its_state(zh_tw, public):
    """Questions, Labels and expected Labels above all; the rest stay as jevbench has them too."""
    differences = [
        f"{t['id']}: {field}"
        for part in PARTS
        for t, source in zip(zh_tw[part], public[part], strict=True)
        for field in sorted(set(t) | set(source))
        if field != "state" and t.get(field, "<missing>") != source.get(field, "<missing>")
    ]

    assert differences == []


def _shape(state):
    """A State with every string replaced: its keys, their order, its structure and other values."""
    if isinstance(state, dict):
        return [(key, _shape(value)) for key, value in state.items()]
    if isinstance(state, list):
        return [_shape(value) for value in state]
    return str if isinstance(state, str) else state


def test_a_json_state_keeps_its_keys_structure_and_non_string_values(zh_tw, public):
    """Only a JSON State's string values are translated; a text State stays text."""
    changed = [
        t["id"]
        for part in PARTS
        for t, source in zip(zh_tw[part], public[part], strict=True)
        if _shape(t["state"]) != _shape(source["state"])
    ]

    assert changed == []
    assert sum(isinstance(t["state"], dict) for t in zh_tw["zh-tw-hard"]) == 35


_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003ffff]")


def _text(state) -> str:
    """Every string in a State, whether it is text or JSON."""
    if isinstance(state, dict):
        return "".join(map(_text, state.values()))
    if isinstance(state, list):
        return "".join(map(_text, state))
    return state if isinstance(state, str) else ""


def _traditional(han: str) -> bool:
    """Whether a Han character is one Traditional-Chinese text uses: Big5, Taiwan's character set,
    has it. Simplified-only forms (们, 这, 软, 数) are not in Big5."""
    try:
        han.encode("big5")
    except UnicodeEncodeError:
        return False
    return True


def test_every_state_is_non_empty_traditional_chinese_text(zh_tw):
    problems = {}
    for tasks in zh_tw.values():
        for t in tasks:
            han = _HAN.findall(_text(t["state"]))
            if not han:
                problems[t["id"]] = "no Chinese text"
            elif not_traditional := sorted({c for c in han if not _traditional(c)}):
                problems[t["id"]] = "not Traditional Chinese: " + "".join(not_traditional)

    assert problems == {}


def test_the_check_on_traditional_chinese_tells_simplified_characters_apart():
    assert all(map(_traditional, "們這說軟資訊帳預設"))
    assert not any(map(_traditional, "们这说软资讯账预设"))
