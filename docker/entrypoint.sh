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

# GB10 unified-memory gotcha: cudaMemGetInfo counts only truly-free RAM, not
# reclaimable page cache, and FreeToken sizes its caches from that number. A
# box with tens of GB in page cache (image pulls, model downloads) will report
# a few GB "free GPU memory" and fail budget planning with "cache budget too
# small". Warn when the gap is large; the host-side fix is
# `sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'` before starting.
read -r _ mem_avail_kb < <(grep MemAvailable /proc/meminfo | awk '{print $1" "$2}') || mem_avail_kb=0
read -r _ mem_free_kb < <(grep MemFree /proc/meminfo | awk '{print $1" "$2}') || mem_free_kb=0
if [ "${mem_avail_kb:-0}" -gt 0 ] && [ $((mem_avail_kb - mem_free_kb)) -gt $((32 * 1024 * 1024)) ]; then
    echo "[entrypoint] WARNING: $(((mem_avail_kb - mem_free_kb) / 1024 / 1024)) GiB of RAM is reclaimable page cache." >&2
    echo "[entrypoint] The CUDA driver does not count it as free, so FreeToken's memory" >&2
    echo "[entrypoint] budget may come out too small. If startup fails with 'cache budget" >&2
    echo "[entrypoint] too small', drop caches on the HOST first:" >&2
    echo "[entrypoint]   sudo sh -c 'sync && echo 3 > /proc/sys/vm/drop_caches'" >&2
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
