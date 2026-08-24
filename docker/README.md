# FreeToken Docker image for NVIDIA DGX Spark (GB10)

A `linux/arm64` container that builds FreeToken from source with a native
**sm_121** kernel cache and serves MoE models through FreeToken's OpenAI- and
Anthropic-compatible APIs.

## Requirements

- NVIDIA DGX Spark (GB10 Grace Blackwell), DGX OS / Ubuntu 24.04 arm64
- NVIDIA driver **r580 series** (avoid r590: known CUDA Graph deadlock on GB10)
- Docker with the NVIDIA Container Toolkit (preinstalled on DGX OS)

## Build

From the repository root (build takes a while — it compiles the sm_121 kernel
cache and downloads the cu130 torch + FlashInfer prebuilt-kernel wheels):

```bash
docker build -f docker/Dockerfile -t freetoken-dgx-spark .
```

## Run

```bash
docker run -d --name freetoken \
  --gpus all --ipc=host --ulimit memlock=-1:-1 \
  -v ~/models:/models \
  -v freetoken-home:/root/.freetoken \
  -e FT_MODEL=/models/Qwen3.6-35B-A3B \
  -e FT_EXTRA_ARGS="--memory-ratio 0.5" \
  -p 1919:1919 \
  freetoken-dgx-spark
```

Or with compose: `FT_MODEL=/models/<model> docker compose -f docker/docker-compose.yml up --build`

Any explicit command is passed to `ft` verbatim:

```bash
# one-shot bandwidth/backend profile (recommended once per box; persists in the
# freetoken-home volume and steers --moe-backend auto)
docker run --rm --gpus all --ipc=host --ulimit memlock=-1:-1 \
  -v freetoken-home:/root/.freetoken freetoken-dgx-spark bench bw
```

Smoke test:

```bash
curl -s http://localhost:1919/health
curl -s http://localhost:1919/v1/models
curl -s http://localhost:1919/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"Say hi"}],"max_tokens":32}'
```

## DGX Spark specifics baked into this image

| Choice | Why |
|---|---|
| `FREETOKEN_KERNEL_CACHE_ARCHES=12.1` at build | GB10 is sm_121; upstream's default arch list stops at 12.0 and would fall back to PTX JIT on every cold start |
| `[fi]` extra only, **no** `[sgl]` | sglang-kernel 0.4.5 aarch64 wheels carry only sm_90/sm_100 binaries; FreeToken's Triton fallback covers sm_121. Native sm_121a rebuild is planned (see `PLAN.md` Phase 3) |
| Runtime image keeps the CUDA **devel** toolchain | tvm-ffi JIT fallback, the GGUF runtime extension, and uncovered FlashInfer variants all need `nvcc`/`g++` at runtime |
| Default `--memory-ratio 0.5` (vs upstream 0.9) | The 128 GB LPDDR5x pool is shared by GPU caches, host expert banks, KV, and the OS; upstream's default assumes separate VRAM and overcommits |
| `--ulimit memlock=-1` | Host expert banks are pinned with `cudaHostRegister`; the Docker default memlock cap breaks large models. Fallback knob: `FREETOKEN_PIN_BUDGET_GB` |

### Model sizing on 128 GB unified memory

| Model | Format | Fits? |
|---|---|---|
| Qwen3.6-35B-A3B | NVFP4 / BF16 | yes |
| gpt-oss-120b | MXFP4 | yes |
| DeepSeek-V4-Flash | MXFP4 (~140 GB pool) | no |
| GLM-5.2 | NVFP4 (433 GB) | no |

### Useful tuning environment variables

- `FT_MODEL`, `FT_HOST`, `FT_PORT`, `FT_EXTRA_ARGS` — entrypoint knobs
- `FREETOKEN_PIN_BUDGET_GB` — cap pinned host-bank bytes (activates the
  lock-CPU/pin-GPU split-residency path) if full pinning misbehaves
- `FREETOKEN_BANK_CUDA_ALLOC=1` — born-pinned `cudaHostAlloc` banks instead of
  mmap+register-after-fill; worth A/B-testing on unified memory
- `FREETOKEN_BENCHBW_PATH` — relocate the bench profile
- `FREETOKEN_KERNEL_CACHE_ARCHES` — rebuild-time GPU arch list
