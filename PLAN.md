# FreeToken on DGX Spark — Fork & Docker Container Plan

**Date:** 2026-08-24
**Goal:** Fork [FreeToken](https://github.com/FlashML-org/FreeToken) and build a Linux Docker container serving MoE LLMs locally on NVIDIA DGX Spark (GB10 Grace Blackwell: 20-core aarch64 CPU + Blackwell GPU sm_121, 128 GB coherent unified LPDDR5x @ ~273 GB/s, CUDA 13, DGX OS 7.x).

---

## 1. Feasibility verdict: **FEASIBLE** (moderate effort, no hard blockers)

FreeToken (arXiv 2608.16157, Berkeley/MIT/UT Austin, Apache-2.0) is an edge-native MoE serving engine built on the SGLang/vLLM substrate. Analysis of the code, the paper, and the dependency ecosystem shows:

**No source-code blocker exists.**
- All x86 SIMD code (`python/freetoken/kernel/csrc/cpu_moe/cpu_moe_ext.cpp`) is `#if CPU_MOE_X86`-guarded with scalar fallbacks for every weight format; memory ordering was already written to be aarch64-correct (the code explicitly cites GH200/Jetson).
- No `-march` flags, no host inline asm, no `platform_machine` markers anywhere in the build. "Linux x86_64" in `docs/install.md` is a documentation claim, not a code constraint.
- Every pinned native dependency has an aarch64 wheel in the exact pinned version: torch 2.11+cu130, triton 3.6.0, apache-tvm-ffi 0.1.13.post3, flashinfer (pure-Python + aarch64 jit-cache wheels on flashinfer.ai/whl/cu130), flashlib 0.3.0 (pure-Python).

**The architecture ports *better* than expected — unified memory is a tailwind, not a headwind.**
- The hot offload path is **not** `cudaMemcpyAsync` over PCIe. It is a GPU kernel dereferencing pinned host memory directly via UVA (`csrc/jit/fast_index_copy.cuh:129-160`). On Spark this read goes over coherent NVLink-C2C at ~200+ GB/s instead of ~31 GB/s PCIe — a ~9× uplift in the engine's single most important mechanism, with **zero code change**.
- The bandwidth-adaptive q\* policy is *measured, never assumed* (`ft bench bw` → profile keyed on GPU name). On Spark it will measure fast "PCIe" (C2C) vs. slow scalar-ARM CPU kernels and `--moe-backend auto` will correctly settle on pure `offload` — exactly what the paper predicts as the degenerate case when transfer bandwidth approaches host bandwidth (§3.2).
- What loses its purpose on unified memory (per the paper): the CPU co-execution branch, prefill double-buffered streaming as a PCIe-hiding trick, and the expert LRU cache as a copy target. What stays valuable: semantic-aware KV/state checkpointing for agentic workloads, FTW fast bootstrap (direct-I/O checkpoint load, no warmup), elastic memory budgeting, CUDA-graph MoE dispatch, and the OpenAI/Anthropic-compatible server.

**The real work items:**
1. Build aarch64 wheels from source (none are published; CI is x86-only) — but `scripts/ci/manylinux-build.sh` already builds inside Docker and is ~90% of the Dockerfile.
2. Add GB10 to the kernel-cache arch list (`FREETOKEN_KERNEL_CACHE_ARCHES="12.1"`; default list in `freetoken-kernel-cache/build_backend.py:99-113` stops at 12.0).
3. `sglang-kernel==0.4.5` aarch64 wheels contain only sm_90/sm_100 binaries → either drop the `[sgl]` extra (documented pure-Triton fallback) or rebuild it for `sm_121a` (solved problem in the SGLang-on-Spark community, e.g. ubehera/sglang-spark, sglang#11658).
4. Re-tune memory budgeting: `engine/cache_budget.py` treats "free VRAM" and host bank RAM as independent pools; on GB10 they are the same 128 GB. Default `--memory-ratio 0.9` double-dips → the most likely OOM failure mode.

**Model reality check on 128 GB unified memory:**

| Model | Format | Expert pool | Fits? |
|---|---|---|---|
| Qwen3.6-35B-A3B | BF16 / NVFP4 | ~70 GB / ~20 GB | ✅ comfortably |
| gpt-oss-120b | MXFP4 | ~63 GB | ✅ |
| GLM-4.7 / mid-size MoE | NVFP4 | varies | ✅ likely |
| DeepSeek-V4-Flash (284B) | MXFP4 | ~140 GB | ❌ over 128 GB (paper needed 140 GB host + VRAM) |
| GLM-5.2 (753B) | NVFP4 | 433 GB | ❌ far out of reach |

Realistic first target: **Qwen3.6-35B-A3B** (paper's reference model), then **gpt-oss-120b**.

---

## 2. Key hardware/software facts (verified 2026-08)

- DGX Spark ships DGX OS 7.x (Ubuntu 24.04 base), CUDA 13.0, driver r580.x. **Avoid driver 590.x** — confirmed CUDA Graph deadlock on GB10 (FreeToken relies heavily on CUDA Graphs).
- NGC arm64 containers ≥ 25.10 support GB10 via sm_120 SASS + compute_120 PTX; native `sm_121a` compilation on-box is preferred for FreeToken's kernel cache.
- PyTorch official cu130 aarch64 wheels run on GB10 via sm_120↔sm_121 family compatibility / PTX JIT (works; occasional heavy-kernel JIT stalls reported — mitigate by pre-warming, or a source-built torch later if needed).
- Triton 3.6 has known `ptxas` failures on `sm_121a` for some exotic kernels (seen with gpt-oss `tile::gather4`) — must smoke-test the Triton fused-MoE paths.
- FreeToken's backend auto-selection already does the right thing on sm_121: attention lands on `fi` (FlashInfer) or `triton` (`fa`/`trtllm` are correctly gated out); NVFP4 expert GEMM lands on `b12x` (gate is `cc >= (12,0)` + CUDA ≥ 13 — sm_121 qualifies) or `triton`.

---

## 3. Plan

### Phase 0 — Fork & scaffolding (½ day)
- [ ] Fork `FlashML-org/FreeToken`; create branch `feat/dgx-spark`.
- [ ] Add `docker/` directory: `Dockerfile`, `docker-compose.yml`, `entrypoint.sh`, `README.md`.
- [ ] Decide build strategy: build **on the Spark itself** (native aarch64, simplest) vs. cross/QEMU on x86 (slow for nvcc; not recommended) vs. cloud arm64 runner (GH Actions `ubuntu-24.04-arm`).

### Phase 1 — Minimal working container (2–3 days)
Multi-stage Dockerfile, conservative kernel choices (no sgl-kernel):

- **Builder stage:** `nvcr.io/nvidia/cuda:13.0-devel-ubuntu24.04` (arm64) — or `pytorch/manylinux2_28_aarch64-builder:cuda13.0` to reuse `scripts/ci/manylinux-build.sh` with `FT_BUILDER_IMAGE` override and `FT_MANYLINUX_RETAG=0` (the retag step hard-fails on its `linux_x86_64` glob otherwise; `manylinux-build.sh:122-127`).
  - `FREETOKEN_KERNEL_CACHE_ARCHES="12.1"` → native sm_121 fatbins, small image, no cold-start PTX JIT.
  - Build `freetoken` + `freetoken-kernel-cache` wheels via `scripts/build-release-wheels.sh` (needs git tree; or `FREETOKEN_BUILD_NO_STAMP=1`).
- **Runtime stage:** `nvcr.io/nvidia/cuda:13.0-devel-ubuntu24.04` — keep the **devel** image (nvcc must stay: tvm-ffi JIT fallback, GGUF runtime extension, FlashInfer JIT for uncovered variants all need it). Python 3.12 venv.
  - Install with the four extra index URLs from `install.sh:211-216` (`--index-strategy unsafe-best-match` under uv): pytorch cu130, sglang cu130, flashinfer.ai/whl + /whl/cu130.
  - Extras: `[fi]` only (FlashInfer). **Skip `[sgl]`** in Phase 1 — its aarch64 wheel lacks sm_12x binaries; FreeToken falls back to pure-Triton kernels by design.
  - `flashinfer-jit-cache` from flashinfer.ai/whl/cu130 (aarch64 cp39-abi3 wheel exists; not on PyPI).
- **Entrypoint:** `ft serve --host 0.0.0.0` (default bind is 127.0.0.1:1919 — must override in a container); volumes for `~/.freetoken` (models, FTW checkpoints, kernel cache, bench profiles); `--gpus all` / NVIDIA Container Toolkit (preinstalled on DGX OS).
- [ ] Smoke test: convert + serve **Qwen3.6-35B-A3B (NVFP4)**, verify `/v1/chat/completions` and `/v1/messages`, run `tests/` subsets (kernels, moe, engine) inside the container.

### Phase 2 — Spark-specific tuning (3–5 days)
- [ ] Run `ft bench bw` in-container; ship the profile via `FREETOKEN_BENCHBW_PATH` on a volume (profiles are keyed on GPU name, so x86 profiles are auto-discarded — good).
- [ ] Confirm `auto` selects `offload` backend + `fi` attention; benchmark `nvfp4_backend` `b12x` vs `triton` (config default is `triton`).
- [ ] **Memory budget:** empirically find safe `--memory-ratio` (start ~0.5–0.6) and explicit `--moe-cache-size` / `--kv-reserve-tokens`; document that expert banks, "VRAM" caches, KV, and the OS all share 128 GB. Keep `FREETOKEN_PIN_BUDGET_GB` in the runbook as a safety valve if `cudaHostRegister` of huge banks misbehaves on LPDDR5x.
- [ ] A/B `FREETOKEN_BANK_CUDA_ALLOC` (born-pinned vs pin-after-fill) — the x86 rationale may not hold on unified memory.
- [ ] Re-measure the `cudaMemcpyBatchAsync` small-bank threshold (`moe/offload_cache.py:16-25` hack targets an H100/CUDA-13.0 bug; may be wrong on GB10).
- [ ] Verify page size assumption: if DGX OS runs a 64 K-page kernel, smoke-test FTW direct-I/O load (`moe/host_banks.py:37` hardcodes 4096 alignment — still correct, but test).
- [ ] Triton sm_121a smoke sweep over the fused-MoE kernels (known ptxas issues on some kernels).
- [ ] Baseline comparison: tok/s and TTFT vs. llama.cpp and Ollama on the same Spark, same models.

### Phase 3 — Optional performance work (1–2 weeks, prioritize by Phase 2 data)
- [ ] **Rebuild sglang-kernel 0.4.5 from source for `sm_121a`** and re-enable `[sgl]` if Triton fallback shows gaps (community recipes exist: sglang#11658, ubehera/sglang-spark).
- [ ] **Unified-memory fast path (upstreamable contribution):** on coherent systems, "cache fill" copies are optional — experts could be read in place. Candidate changes: force q\*→m degenerate mode explicitly; optionally bypass the GPU expert cache entirely when `B_P ≈ B_H` (new "unified" MoE backend); skip prefill double-buffering (prefill becomes compute-bound anyway). The paper (§3.2) explicitly anticipates the degenerate case; the codebase does not yet have a pure-GPU-from-unified-memory mode.
- [ ] Contribute a GB10 Triton fused-MoE tuning config (`python/freetoken/moe/configs/` currently has only an H100 entry).
- [ ] If ever using the `cpu`/`hybrid` backends: NEON/SVE2 kernels for `cpu_moe_ext.cpp` (currently scalar on ARM) + big.LITTLE-aware thread pinning (X925 vs A725; `physical_core_cpus()` is SMT-aware but not cluster-aware). **Likely unnecessary** — on unified memory the CPU branch contends with the GPU for the same 273 GB/s and probably hurts.
- [ ] gpt-oss-120b (MXFP4) bring-up as the second model.

### Phase 4 — CI & packaging (2–3 days)
- [ ] GH Actions aarch64 lane (`ubuntu-24.04-arm` runners or self-hosted Spark) mirroring `nightly-wheels.yml`; fix the `linux_x86_64` filename globs in `scripts/publish-wheels.sh:38` and the nightly staleness grep.
- [ ] Publish the container image (GHCR, `linux/arm64`), tagged with CUDA/driver compatibility.
- [ ] Runbook: driver pin (r580, avoid 590), model download/conversion to FTW, memory sizing table per model, agent integration (Claude Code / Codex via the Anthropic/OpenAI APIs).

---

## 4. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Default memory budgeting double-counts the shared 128 GB → OOM | High, certain | Phase 2 tuning; conservative defaults baked into the container entrypoint |
| Triton 3.6 ptxas failures on sm_121a in some fused kernels | Medium | Smoke sweep early; per-kernel fallback; track Triton upstream |
| PTX-JIT stalls in torch cu130 wheels on GB10 | Medium | Pre-warm in entrypoint; native `12.1` kernel-cache build removes it for FreeToken's own kernels |
| Driver 590 CUDA Graph deadlock (FreeToken is graph-heavy) | High if hit | Pin r580 in docs and image labels |
| DSV4-Flash / GLM-5.2 don't fit in 128 GB | Fact, not risk | Scope to Qwen3.6 / gpt-oss-120b class; note multi-Spark (ConnectX link) as future exploration |
| flashinfer jit-cache coverage gaps for sm_121 variants | Low | nvcc kept in runtime image → JIT covers gaps |

## 5. Effort estimate

- Minimal working container: **~1 week**
- Tuned, benchmarked, documented v1: **~2–3 weeks**
- With sm_121a sgl-kernel rebuild + unified-memory fast path + CI: **~4–5 weeks**

## 6. Progress log

**2026-08-24 — Phase 0+1 complete, Phase 2 mostly complete** (executed natively on a DGX Spark: GB10, driver 580.173.02, CUDA 13.0, 120 GB, DGX OS).

Phase 0/1: fork `thupalo/FreeToken` branch `feat/dgx-spark`; image `freetoken-dgx-spark` (19.2 GB) builds in ~12 min with native sm_121 SASS (verified via cuobjdump); serving Qwen3.6-35B-A3B-NVFP4 works via OpenAI + Anthropic APIs.

Phase 2 configuration sweep (Qwen3.6-35B-A3B-NVFP4, median of 3 warm runs):

| config | startup | decode tok/s | TTFT | prefill tok/s (4k prompt) |
|---|---|---|---|---|
| ratio 0.5, triton (default) | 47 s | **63.5** | 0.35 s | 1062 |
| ratio 0.7, triton | 50 s | 63.7 | 0.35 s | 1065 |
| ratio 0.85, triton | 129 s | 63.2 | 0.36 s | 1110 |
| ratio 0.7, b12x (`--nvfp4-backend flashinfer`) | **fails** | — | — | — |
| ratio 0.7, + OpenAI triton_kernels router | 61 s | 62.5 | 0.35 s | 813 |
| ratio 0.7, `FREETOKEN_BANK_CUDA_ALLOC=1` | 50 s | 62.9 | 0.36 s | 1059 |

Findings:
- **Defaults win.** Decode is memory-bandwidth-bound at ~63 tok/s and invariant to cache size, bank allocation, and router — the unified-memory prediction confirmed empirically (cache hit and miss read the same LPDDR5x). `--memory-ratio 0.5` keeps fastest startup and most free RAM. 63 tok/s sits between the paper's RTX 4090 (42.9) and RTX 5090 (76.7) numbers for this model.
- **b12x backend broken on GB10 for this model**: `OverflowError: Value overflow: 2684354560 exceeds range of l` in `nvidia_cutlass_dsl` `build_memref_desc` via `flashinfer/fused_moe/cute_dsl/blackwell_sm12x/moe_w4a16_kernel.py` — a ~2.5 GiB expert-bank tensor descriptor overflows a 32-bit field that smaller-VRAM RTX 50 setups never hit. Upstream (flashinfer/cutlass-dsl) bug; `triton` NVFP4 backend is the backend of record.
- **`ft bench bw` on GB10**: C2C "PCIe" gather 78–85 GB/s, CPU STREAM 103 GB/s, scalar-ARM CPU-MoE 9–34 GB/s → auto-picks `offload` for every format, as designed.
- **Tests**: `tests/kernels` + `tests/moe` in-container: 303 passed, 4 skipped, 0 failed — no Triton sm_121a ptxas issues in FreeToken's kernel set.
- **Ops gotchas** (documented in docker/README.md): GB10 `cudaMemGetInfo` ignores reclaimable page cache → "cache budget too small" after big downloads (fix: drop_caches); bench-bw profile persisted via `XDG_CACHE_HOME` in the volume; page size is 4 K (FTW O_DIRECT assumption holds); PyPI's `triton-kernels` package is NOT OpenAI's `triton_kernels` (name collision — install from the triton repo subdirectory if ever needed).

llama.cpp head-to-head (same box, same prompts; llama.cpp on MXFP4_MOE GGUF, full GPU offload): FreeToken 63.5 vs llama.cpp 59.4 tok/s decode (+7%), but llama.cpp wins prefill 2357 vs 1062 tok/s (2.2×) and short-prompt TTFT 0.073 vs 0.35 s. The paper's 1.8–2.3× decode advantage on PCIe rigs collapses on unified memory, as predicted; FreeToken's differentiators here are agentic semantic caching, elastic memory, and the dual API surface. Prefill is the optimization target.

### Speculative decoding assessment (2026-08-24)

Feasible and the highest-leverage Phase 3 item: decode is bandwidth-bound, and verifying k drafted tokens costs roughly the same weight traffic as generating one, so accept-length multiplies tok/s almost directly.

- **Method choice: native MTP first.** The NVFP4 checkpoint already ships a 1-layer MTP head (`mtp_num_hidden_layers=1`) that FreeToken currently *drops at load* (`python/freetoken/models/qwen3_5_moe/weight.py:37-38,136,608`). Published benchmarks show native MTP beating EAGLE-3 and DFlash heads for this exact model (accept ~1.85–1.9 at 1 draft token, 85–93% acceptance); vLLM's DGX Spark recipe uses MTP with `num_speculative_tokens: 3`. Reference points: vLLM+NVFP4+MTP on Spark = 55.9 tok/s single-user (FreeToken already does 63.5 *without* spec decode); llama.cpp+MTP on Spark (27B dense) ≈ 2.2× single-stream.
- **DSpark** (DeepSeek, arXiv 2607.05147, semi-autoregressive block drafter with confidence-scheduled verification; in SGLang/vLLM/llama.cpp; a trained drafter for this exact model exists: RedHatAI/Qwen3.6-35B-A3B-speculator.dspark) is the only method with a credible shot at beating MTP, but honest single-user GB10 numbers are 1.08–1.3× over MTP on novel code (2.2× only on repetitive text), and prose can regress. Do MTP first, DSpark as a later experiment on the same verification infra.
- **"DSpark2" does not exist** — no paper, repo, or vendor feature by that name (likely conflated with DFlash-V2 draft checkpoints or the DeepSpec training repo).
- **Engineering scope in FreeToken (est. 2–4 weeks for MTP-1/2, greedy-first):** (1) stop dropping `mtp.*` weights and map the MTP layer's experts into FTW banks (it is itself a MoE layer); (2) draft loop after each target step; (3) multi-token verification step — currently the engine samples exactly 1 token/seq/step (`engine/engine.py:927-931`) with CUDA graphs at bs 1/2/4, so verification needs new graph shapes (bs × draft-len) and FlashInfer's multi-token decode path; (4) rollback: paged KV truncation is easy for the 10 full-attention layers, but the 30 gated-delta linear-attention layers need per-step recurrent-state checkpointing — FreeToken's semantic-anchor state checkpoint machinery is the natural base, extended to per-token granularity during verification windows; (5) accept/reject sampling. Realistic target: ~63 → ~95–120 tok/s single-user at accept ≈ 1.8–2.
- Caveat from llama.cpp-on-Spark reports: MTP can hurt batched throughput at concurrency ≥ 4 — gate it on batch size.

### Speculative decoding EVALUATION results (2026-08-24, empirical)

Measured on this box with the official llama.cpp `server-cuda` arm64 image
(build 10603), Qwen3.6-35B-A3B MXFP4_MOE GGUF with embedded MTP head
(unsloth MTP-GGUF), full GPU offload, greedy decoding, median of 3 × 256-token
runs per content type. DSpark drafter: williamliao Q8_0 GGUF (556 MB,
RedHatAI-lineage). Production FreeToken container stopped during runs.

| config | repetitive | code | prose | acceptance / mean accepted len |
|---|---|---|---|---|
| baseline (no spec) | 62.7 | 62.7 | 62.9 | — |
| MTP, 2 drafts | 91.1 | 82.9 | 94.0 | 98.3% / 2.97 |
| **MTP, 3 drafts** | **112.2** | **95.9** | **109.3** | 96.9% / 3.91 |
| MTP, 3 drafts + p-min 0.75 | 81.5 | 76.3 | 108.7 | 100% / 4.00 (too few drafts issued) |
| DSpark (dedicated drafter) | 115.5 | 99.2 | 114.6 | 93.0% / 3.79 |

Conclusions:
- **Go for the FreeToken MTP implementation.** The model's built-in MTP head at
  3 drafts delivers 1.5–1.8× (avg ~1.7×) over baseline and lands within 3% of
  DSpark — with no extra drafter model, no extra weight traffic, and weights we
  already ship (currently dropped at load). Acceptance is far above published
  figures (97–98% vs 85–93%), so 4–5 drafts may push further.
- The confidence floor (`p-min 0.75`) *hurts* on GB10 — the head is
  well-calibrated; gating drafts wastes bandwidth-amortization opportunity.
- DSpark works out of the box in llama.cpp and is the ceiling reference
  (~1.6–1.8×); as a FreeToken feature it would add a separate drafter pipeline
  for ≤3% gain — not worth it before MTP ships.
- Current llama.cpp (Aug build) baseline is 62.7–62.9 tok/s — it caught up with
  FreeToken (63.5) since the April build (59.4). With MTP it reaches ~96–112
  tok/s TODAY, which FreeToken cannot match until MTP is implemented; this is
  now the top competitive gap on this platform.
- Validated target for FreeToken MTP: **~100–115 tok/s** single-user decode.

**gpt-oss-120b bring-up (2026-08-24): works.** First-serve FTW conversion of the 61 GB MXFP4 checkpoint OK; decode 24.8 tok/s median (offload backend, triton attention, swa_radix cache). Second validated model for the container.

### M4 progress log (2026-08-24/25)

- **Profile of an eager verify step (bs=1, k=1)**: 22 ms GPU / 31 ms CPU — host-bound. GPU
  +6 ms vs a decode step from small-M kernel variants (`aten::_scaled_mm` W8A8 at M=2 6.5 ms,
  cutlass FMHA prefill wrapper ~4.6 ms, nvfp4 gemm-vs-gemv); host: `ChunkGatedDeltaRuleFunction`
  ~8 ms across 30 GDN layers (1.4 ms of GPU work). Eager 1-token decode = 53.2 tok/s
  (18.8 ms) vs 62.7 graphed → launch overhead is only ~3 ms; **kernel choice first, graphs last.**
- GDN verify via the FLA *decode* kernel (varlen T=2) + multi-token conv update: step
  35.4 → 26.1 ms, kernels pass parity offline (conv exact, GDN ≤ bf16 noise, bs=1/T=2,3) but
  produce garbage in situ (mixed/decode paths identical garbage, chunk path fine) — root
  cause under investigation with an in-situ tensor dump (FREETOKEN_MTP_GDN_PATH /
  FREETOKEN_MTP_GDN_DUMP debug knobs).
- **GPU+CPU `hybrid` MoE backend measured: 26.3 tok/s vs 62.7 offload** (scalar-ARM CPU
  branch becomes the per-layer critical path; contends for the same LPDDR5x). Bandwidth-teaming
  remains a research bet gated on a concurrent-bandwidth ceiling test (see discussion).
- **sparkrun integration works**: recipe `/mnt/shared/spark/freetoken/freetoken-qwen3.6-nvfp4.yaml`
  (runtime: vllm, `executor_config.entrypoint: ""`, HF-cache model id, cache env redirected
  into /cache/runtime). First official-harness result on spark2 (default profile, TP1):
  **tg32 = 61.0 ± 0.5 tok/s** (matches direct probes; tightest std on the board), pp2048 bogus
  (2.2M t/s) because FreeToken streamed an empty role chunk before prefill (ttfr 6 ms) — fixed:
  role chunk now goes out with the first token (commit 1f2c0c8); `/v1/models` returns 503 until
  serving (sparkrun's readiness probe raced the load). Bar to beat on this board: tg32 142.8
  (vLLM marlin+MTP), pp2048 6382 (vLLM 0.27.2 b12x+MTP).

Remaining Phase 3 candidates otherwise unchanged. The b12x int32 overflow was reported upstream: [flashinfer#4706](https://github.com/flashinfer-ai/flashinfer/issues/4706) (checked distinct from #2776/#3383 before filing).

## 7. Sources

- Repo: cloned at `FreeToken/` (commit `bd372b6`); paper: `2608.16157.pdf`
- SGLang on Spark: github.com/sgl-project/sglang/issues/11658; vLLM sm_121 tracking: vllm#31128, #36821
- DGX Spark stack: docs.nvidia.com/dgx/dgx-spark/ngc.html; PyTorch cu130 aarch64 wheels: download.pytorch.org/whl/cu130
