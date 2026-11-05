# 11: Dev container

**What to build:** The developer opens the repo in a dev container (with OrbStack) and gets a reproducible Linux environment. In it they can run every test tier, including the real models, and serve and benchmark either Decider. On a host with a GPU, the same container uses the GPU.

**Blocked by:** 10 (Docker image and compose)

**Status:** resolved

- [x] A dev container configuration:
  - built from the Dockerfile's `dev` target;
  - uses the GPU when the host has one and works without it;
  - shares the Hugging Face cache volume with compose;
  - runs the uv sync on create;
  - recommends the Python and Ruff editor extensions.
- [x] Inside the container on the Mac (CPU):
  - the fast test tiers pass;
  - the `real_model` tier passes;
  - either Decider can be served and benchmarked.
- [x] The README has a dev-container section. It notes that MPS is only available natively, not in the container.

## Comments

- 2026-11-05 (implement-spec): Merged in 82063a6. Verified by reproducing the devcontainer config with plain docker (no devcontainer CLI): fast and real_model tiers pass on CPU, and both Deciders were served and benchmarked. GPU path unverified. Integration branch `local-jev-v1`; code-review fixes merged in ce0131c and 2338b2f.
