# syntax=docker/dockerfile:1

# Two targets on one base:
#   runtime  the deployable server (compose.yaml builds it): locked runtime dependencies only.
#   dev      the dev container's image: dev dependencies too, plus git for `local-jev bench`;
#            the source is bind-mounted over /app and installed by `uv sync` on create.
# Model weights are not in either image: the server downloads them at its pinned revision into
# the Hugging Face cache, HF_HOME=/data/hf, which compose mounts as a named volume.
# torch comes from uv.lock: CUDA 12.8 builds on linux/amd64, CPU builds on linux/arm64.

FROM ghcr.io/astral-sh/uv:0.13.6@sha256:af22a52f5eafef3d78e551599d405c02df61919d18028f0cea8eda93bc9986d0 AS uv

FROM ubuntu:24.04 AS base
COPY --from=uv /uv /uvx /bin/
ENV UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    UV_PYTHON_PREFERENCE=only-managed \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/data/hf
# gcc and the C library headers stay in every image: Triton compiles kernels at run time on CUDA
# (LFM3's conv kernel), against the managed Python's own headers.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*
# Installed as root into a world-readable directory; the venv's python links into it.
RUN uv python install 3.12
# ubuntu:24.04 already has user "ubuntu" with UID 1000. A fresh named volume mounted on /data/hf
# takes this directory's owner, so the non-root server can download weights into it.
RUN useradd --create-home --uid 10001 --user-group app \
    && mkdir -p /data/hf \
    && chown app:app /data/hf
WORKDIR /app

FROM base AS dev
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
# The dev container runs `uv sync` as app, into this venv.
RUN mkdir -p /opt/venv && chown app:app /app /opt/venv
USER app
RUN --mount=type=cache,target=/home/app/.cache/uv,uid=10001,gid=10001 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project

FROM base AS runtime
# Dependencies in their own layer, rebuilt only when the lockfile changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable
USER app
EXPOSE 8000
# Serves d1 unless told otherwise: `docker run … local-jev:runtime --decider lfm3`.
ENTRYPOINT ["local-jev", "serve", "--host", "0.0.0.0", "--port", "8000"]
