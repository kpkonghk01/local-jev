"""What a dev container tool (VS Code Dev Containers, the `devcontainer` CLI) would build and run
from `.devcontainer/devcontainer.json`.

Whether the fast and `real_model` tiers pass inside it, and whether either Decider serves and
benchmarks there, is checked by hand (README, dev container section).
"""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / ".devcontainer"

needs_compose = pytest.mark.skipif(
    shutil.which("docker") is None
    or subprocess.run(["docker", "compose", "version"], capture_output=True).returncode != 0,
    reason="needs the docker CLI with compose",
)


def _config() -> dict:
    """devcontainer.json as the tools read it: JSON with comments."""
    text = (CONFIG_DIR / "devcontainer.json").read_text(encoding="utf-8")
    no_comments = re.sub(
        r'"(?:\\.|[^"\\])*"|//[^\n]*|/\*.*?\*/',
        lambda m: m.group(0) if m.group(0).startswith('"') else "",
        text,
        flags=re.S,
    )
    return json.loads(no_comments)


def _compose_project() -> dict:
    """compose.yaml as `docker compose` resolves it, with every Decider's profile on."""
    shell = {k: v for k, v in os.environ.items() if not k.startswith(("LOCAL_JEV_", "COMPOSE_"))}
    done = subprocess.run(
        ["docker", "compose", "--project-directory", str(ROOT), "-f", "compose.yaml"]
        + ["--env-file", os.devnull, "--profile", "d1", "--profile", "lfm3"]
        + ["config", "--format", "json"],
        cwd=ROOT,
        env=shell,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_it_is_built_from_the_dockerfiles_dev_target():
    build = _config()["build"]

    assert (CONFIG_DIR / build["dockerfile"]).resolve() == ROOT / "Dockerfile"
    assert (CONFIG_DIR / build["context"]).resolve() == ROOT
    assert build["target"] == "dev"


@needs_compose
def test_it_mounts_the_same_whole_hugging_face_cache_volume_as_compose():
    project = _compose_project()
    ((volume, hf_home),) = {
        (project["volumes"][v["source"]]["name"], v["target"])
        for service in project["services"].values()
        for v in service["volumes"]
    }

    assert {"type": "volume", "source": volume, "target": hf_home} in _config()["mounts"]


def test_it_uses_a_gpu_when_the_host_has_one_and_starts_without_one():
    config = _config()

    # "optional": the tool adds the host's GPUs only when Docker reports the NVIDIA runtime.
    assert config["hostRequirements"]["gpu"] == "optional"
    assert [arg for arg in config.get("runArgs", []) if arg.startswith("--gpus")] == []


def test_on_create_it_installs_the_repo_mounted_at_app_from_the_lockfile():
    config = _config()
    mount = dict(item.split("=", 1) for item in config["workspaceMount"].split(","))

    assert mount == {"type": "bind", "source": "${localWorkspaceFolder}", "target": "/app"}
    assert config["workspaceFolder"] == "/app"
    assert config["postCreateCommand"] == "uv sync --frozen"


def test_it_recommends_the_python_and_ruff_editor_extensions():
    extensions = _config()["customizations"]["vscode"]["extensions"]

    assert {"ms-python.python", "charliermarsh.ruff"} <= set(extensions)
