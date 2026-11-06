"""The lockfile installs the intended torch build on each platform local-jev runs on.

Read from `uv.lock` itself, the file every `uv sync --frozen` (native, image, dev container) obeys.
"""

import tomllib
from pathlib import Path

import pytest

LOCK = tomllib.loads((Path(__file__).parents[1] / "uv.lock").read_text(encoding="utf-8"))

MAC = "platform_machine == 'arm64' and sys_platform == 'darwin'"
GPU_HOST = "platform_machine == 'x86_64' and sys_platform == 'linux'"
ARM_CONTAINER = "platform_machine == 'aarch64' and sys_platform == 'linux'"

PYPI = "https://pypi.org/simple"
CU128 = "https://download.pytorch.org/whl/cu128"
CPU = "https://download.pytorch.org/whl/cpu"


def _locked(name: str, platform: str) -> dict:
    """The one locked version of `name` that installs on `platform`."""
    entries = [p for p in LOCK["package"] if p["name"] == name]
    (entry,) = [p for p in entries if platform in p.get("resolution-markers", [platform])]
    return entry


@pytest.mark.parametrize(
    ("platform", "torch", "torchvision", "index"),
    [
        (MAC, "2.11.0", "0.26.0", PYPI),
        (GPU_HOST, "2.11.0+cu128", "0.26.0+cu128", CU128),
        (ARM_CONTAINER, "2.11.0+cpu", "0.26.0+cpu", CPU),
    ],
    ids=["mac-mps", "linux-x86_64-cuda-12.8", "linux-aarch64-cpu"],
)
def test_each_platform_gets_torch_2_11_from_its_own_index(platform, torch, torchvision, index):
    locked = {name: _locked(name, platform) for name in ("torch", "torchvision")}

    assert {name: (p["version"], p["source"]["registry"]) for name, p in locked.items()} == {
        "torch": (torch, index),
        "torchvision": (torchvision, index),
    }


def test_the_arm64_container_installs_no_cuda_libraries():
    dependencies = {d["name"] for d in _locked("torch", ARM_CONTAINER).get("dependencies", [])}

    assert [d for d in dependencies if d.startswith(("nvidia-", "cuda-", "triton"))] == []
