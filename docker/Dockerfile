# syntax=docker/dockerfile:1
#
# FreeToken serving container for NVIDIA DGX Spark (GB10 Grace Blackwell).
#
#   linux/arm64 (aarch64) - CUDA 13.0 - GPU arch sm_121 - driver r580+
#
# Build from the repository root:
#   docker build -f docker/Dockerfile -t freetoken-dgx-spark .
#
# The kernel cache is compiled natively for sm_121 only (see
# FREETOKEN_KERNEL_CACHE_ARCHES below), so this image is DGX-Spark-specific:
# no cold-start PTX JIT, small fatbins. The runtime stage keeps the full CUDA
# devel toolchain because three subsystems JIT-compile on demand: the tvm-ffi
# kernel fallback, the GGUF runtime extension (Gemma-4), and FlashInfer
# variants not covered by the prebuilt jit-cache wheel.
#
# Phase 1 deliberately skips the [sgl] extra: sglang-kernel 0.4.5 aarch64
# wheels ship only sm_90/sm_100 binaries, and FreeToken's documented fallback
# (pure-Triton kernels + 'fi'/'triton' attention) covers sm_121. A native
# sm_121a sglang-kernel rebuild is Phase 3 work.

ARG CUDA_TAG=13.0.2-devel-ubuntu24.04

# ---------------------------------------------------------------- builder --
FROM nvcr.io/nvidia/cuda:${CUDA_TAG} AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl binutils ninja-build build-essential \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV CUDA_HOME=/usr/local/cuda
WORKDIR /src
COPY . /src

# Build venv: build-release-wheels.sh runs `uv build --no-build-isolation`, so
# every build-system requirement (of both the runtime wheel and the
# kernel-cache wheel) must be preinstalled here. torch resolves from the cu130
# index; unsafe-best-match is required because that index mirrors stale copies
# of common deps (see pyproject.toml).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/build-venv --python 3.12 && \
    uv pip install --python /opt/build-venv/bin/python \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        "setuptools>=77" wheel ninja \
        "torch>=2.11,<2.12" "apache-tvm-ffi==0.1.13.post3"

# GB10 (DGX Spark) is compute capability 12.1. The default arch list in
# freetoken-kernel-cache/build_backend.py stops at 12.0; overriding it here is
# the intended mechanism (build-release-wheels.sh prints a loud warning about
# the narrowed fatbin -- expected and intentional for this platform image).
ENV FREETOKEN_KERNEL_CACHE_ARCHES="12.1" \
    FREETOKEN_BUILD_NO_STAMP=1 \
    FREETOKEN_BUILD_PYTHON=/opt/build-venv/bin/python \
    FREETOKEN_BUILD_OUT_DIR=/wheels

RUN bash scripts/build-release-wheels.sh

# ---------------------------------------------------------------- runtime --
FROM nvcr.io/nvidia/cuda:${CUDA_TAG} AS runtime

LABEL org.opencontainers.image.title="FreeToken (DGX Spark)" \
      org.opencontainers.image.description="FreeToken MoE serving engine built for NVIDIA GB10 (sm_121, aarch64, CUDA 13.0)" \
      org.opencontainers.image.source="https://github.com/thupalo/FreeToken" \
      ai.freetoken.platform="dgx-spark-gb10" \
      ai.freetoken.cuda-arch="sm_121" \
      ai.freetoken.driver-note="requires r580 series; r590 has a known CUDA Graph deadlock on GB10"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates curl build-essential ninja-build \
        python3.12 python3.12-venv python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/
COPY --from=builder /wheels /wheels

# Install the freshly built wheels plus the FlashInfer prebuilt-kernel wheels.
# Index args mirror install.sh; the sglang index is omitted along with [sgl].
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /opt/freetoken --python 3.12 && \
    RT_WHL="$(ls /wheels/freetoken-*.whl)" && \
    KC_WHL="$(ls /wheels/freetoken_kernel_cache-*.whl)" && \
    uv pip install --python /opt/freetoken/bin/python \
        --index-strategy unsafe-best-match \
        --extra-index-url https://download.pytorch.org/whl/cu130 \
        --extra-index-url https://flashinfer.ai/whl \
        --extra-index-url https://flashinfer.ai/whl/cu130 \
        "${RT_WHL}[fi]" flashinfer-cubin flashinfer-jit-cache "${KC_WHL}"

ENV PATH="/opt/freetoken/bin:${PATH}" \
    CUDA_HOME=/usr/local/cuda

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Model checkpoints (read-only is fine) and FreeToken state (FTW conversions,
# kernel JIT cache, bench-bw profiles) live on volumes so they survive
# container replacement.
VOLUME ["/models", "/root/.freetoken"]
EXPOSE 1919

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
