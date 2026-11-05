# 10: Docker image and compose

**What to build:** The developer can run either Decider in a container on their Mac (OrbStack, CPU). An operator can run the same services on a Linux NVIDIA GPU host by adding one override file. Model weights live in a persistent volume at the pinned revision, not in the image.

**Blocked by:** 04 (jevbench runner), 05 (LFM3 Option-ID Decider)

**Status:** ready-for-agent

- [ ] Torch comes from the PyTorch CUDA 12.8 index on linux/x86_64, through an explicit index and platform markers, and from the default index elsewhere.
  - The lockfile is regenerated.
  - Native Mac use (MPS) still works.
- [ ] A multi-stage Dockerfile with a `runtime` target:
  - `ubuntu:24.04` base, the uv binary from a pinned image, uv-managed Python 3.12;
  - a frozen sync without dev dependencies;
  - a non-root user;
  - gcc kept, because Triton compiles kernels at run time.
- [ ] A base compose file:
  - one service per Decider, each behind its own profile, so only the selected one starts;
  - a shared Hugging Face cache volume;
  - model id and revision set through the environment;
  - an optional bearer token;
  - a health check on the health endpoint.
  - It works on OrbStack.
- [ ] A GPU override file adds NVIDIA device reservations and is documented as unverified.
- [ ] Verified on the Mac: the d1 profile comes up healthy on OrbStack, and the bench runner's 10-Task smoke run against it gives 10 valid answers. The image builds for linux/arm64.
- [ ] Checked for linux/amd64: the image builds, through emulation if that is practical. Otherwise, the dependency resolution for that platform is checked.
- [ ] The README has a compose section covering CPU and GPU.
