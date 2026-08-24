#!/usr/bin/env bash
# FreeToken container entrypoint (DGX Spark).
#
# With no arguments: serve $FT_MODEL on 0.0.0.0:1919 (overridable via
# FT_HOST/FT_PORT/FT_EXTRA_ARGS). With arguments: pass them to `ft` verbatim,
# so `docker run ... freetoken-dgx-spark bench bw` etc. work as expected.
set -euo pipefail

# Fail early and loudly when the GPU is not mapped into the container.
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "[entrypoint] ERROR: no NVIDIA GPU visible in this container." >&2
    echo "[entrypoint] Run with '--gpus all' (docker run) or 'gpus: all' (compose)." >&2
    exit 1
fi

# Pinning the host expert banks (cudaHostRegister/mlock) needs an unlimited
# memlock rlimit; Docker's default cap breaks large-model bring-up in ways
# that surface late. Warn instead of failing: tiny models can still work.
memlock="$(ulimit -l || echo 0)"
if [ "$memlock" != "unlimited" ]; then
    echo "[entrypoint] WARNING: RLIMIT_MEMLOCK is '$memlock' (not unlimited)." >&2
    echo "[entrypoint] Add '--ulimit memlock=-1:-1' or the compose ulimits block," >&2
    echo "[entrypoint] or set FREETOKEN_PIN_BUDGET_GB to cap pinned bank bytes." >&2
fi

if [ "$#" -eq 0 ]; then
    if [ -z "${FT_MODEL:-}" ]; then
        echo "[entrypoint] ERROR: FT_MODEL is not set." >&2
        echo "[entrypoint] Set FT_MODEL to a model path (e.g. /models/Qwen3.6-35B-A3B)" >&2
        echo "[entrypoint] or pass an explicit 'ft' command line." >&2
        exit 1
    fi
    # shellcheck disable=SC2086 -- FT_EXTRA_ARGS is intentionally word-split
    set -- serve \
        --model "$FT_MODEL" \
        --host "${FT_HOST:-0.0.0.0}" \
        --port "${FT_PORT:-1919}" \
        ${FT_EXTRA_ARGS:-}
fi

exec ft "$@"
