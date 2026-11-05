# 11: Dev container

**What to build:** The developer opens the repo in a dev container (with OrbStack) and gets a reproducible Linux environment. In it they can run every test tier, including the real models, and serve and benchmark either Decider. On a host with a GPU, the same container uses the GPU.

**Blocked by:** 10 (Docker image and compose)

**Status:** ready-for-agent

- [ ] A dev container configuration:
  - built from the Dockerfile's `dev` target;
  - uses the GPU when the host has one and works without it;
  - shares the Hugging Face cache volume with compose;
  - runs the uv sync on create;
  - recommends the Python and Ruff editor extensions.
- [ ] Inside the container on the Mac (CPU):
  - the fast test tiers pass;
  - the `real_model` tier passes;
  - either Decider can be served and benchmarked.
- [ ] The README has a dev-container section. It notes that MPS is only available natively, not in the container.
