"""What `docker compose` would run from compose.yaml (and the GPU override), read through
`docker compose config`. Skipped where the docker CLI with compose is not installed.

Whether the containers actually come up healthy is checked by hand (README, compose section).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None
    or subprocess.run(["docker", "compose", "version"], capture_output=True).returncode != 0,
    reason="needs the docker CLI with compose",
)


def _compose(*args: str, gpu: bool = False, profiles: tuple[str, ...] = (), **env: str) -> dict:
    """The project as compose resolves it, with only `env` set from the operator's shell."""
    files = ["-f", "compose.yaml"] + (["-f", "compose.gpu.yaml"] if gpu else [])
    shell = {k: v for k, v in os.environ.items() if not k.startswith(("LOCAL_JEV_", "COMPOSE_"))}
    done = subprocess.run(
        ["docker", "compose", "--project-directory", str(ROOT), *files, "--env-file", os.devnull]
        + [arg for p in profiles for arg in ("--profile", p)]
        + ["config", "--format", "json", *args],
        cwd=ROOT,
        env={**shell, **env},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _services(*profiles: str, gpu: bool = False) -> list[str]:
    return sorted(_compose(gpu=gpu, profiles=profiles).get("services", {}))


@pytest.mark.parametrize("gpu", [False, True], ids=["cpu", "gpu"])
def test_no_decider_starts_without_a_profile(gpu):
    assert _services(gpu=gpu) == []


@pytest.mark.parametrize("decider", ["d1", "lfm3"])
def test_a_profile_starts_its_decider_alone(decider):
    assert _services(decider) == [decider]


def test_both_deciders_share_one_hugging_face_cache_volume():
    services = _compose(profiles=("d1", "lfm3"))["services"]

    mounts = {
        name: [(v["type"], v["source"], v["target"]) for v in s["volumes"]]
        for name, s in services.items()
    }
    (d1_cache,) = mounts["d1"]
    assert d1_cache[0] == "volume"
    assert mounts == {"d1": [d1_cache], "lfm3": [d1_cache]}


@pytest.mark.parametrize("decider", ["d1", "lfm3"])
def test_the_operator_picks_model_revision_and_token_through_the_environment(decider):
    prefix = f"LOCAL_JEV_{decider.upper()}_"
    project = _compose(
        profiles=(decider,),
        **{f"{prefix}MODEL": "acme/other-2B", f"{prefix}REVISION": "0123abc"},
        LOCAL_JEV_TOKEN="s3cret",
    )
    environment = project["services"][decider]["environment"]

    assert {
        k: environment[k] for k in ("LOCAL_JEV_MODEL", "LOCAL_JEV_REVISION", "LOCAL_JEV_TOKEN")
    } == {
        "LOCAL_JEV_MODEL": "acme/other-2B",
        "LOCAL_JEV_REVISION": "0123abc",
        "LOCAL_JEV_TOKEN": "s3cret",
    }


def test_the_gpu_override_reserves_the_hosts_nvidia_gpus_for_each_decider():
    services = _compose(gpu=True, profiles=("d1", "lfm3"))["services"]

    devices = {
        name: s["deploy"]["resources"]["reservations"]["devices"] for name, s in services.items()
    }
    nvidia_gpus = [{"driver": "nvidia", "count": -1, "capabilities": ["gpu"]}]
    assert devices == {"d1": nvidia_gpus, "lfm3": nvidia_gpus}


def test_without_the_override_no_decider_asks_for_a_gpu():
    services = _compose(profiles=("d1", "lfm3"))["services"]

    assert [name for name, s in services.items() if "deploy" in s] == []
