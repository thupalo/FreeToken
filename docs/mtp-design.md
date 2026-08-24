# MTP speculative decoding for FreeToken — design (feat/mtp-spec-decode)

Target: Qwen3.5/3.6 MoE family on DGX Spark first; mechanism kept model-generic
where cheap. Validated goal: ~100–115 tok/s single-user decode (vs 63.5 today);
empirical ceiling measured via llama.cpp MTP-3 on the same box: 96–112 tok/s at
96.9% draft acceptance (see PLAN.md).

## Head structure (from the NVFP4 checkpoint)

`mtp_num_hidden_layers=1`, `mtp_use_dedicated_embeddings=false`:

- combiner: `h' = fc(concat(pre_fc_norm_embedding(embed(tok)), pre_fc_norm_hidden(h)))`
- one **full-attention** decoder layer (`mtp.layers.0.self_attn.*` with q/k_norm —
  same shape as `Qwen3_5Attention`; NOT gated-delta) + MoE mlp (256 NVFP4 routed
  experts + shared expert + gate) — same shape as `Qwen3_5MoE`
- `mtp.norm`, then the **shared** lm_head/embeddings of the target model

## Key design decisions

1. **MTP experts go through the offload cache as bank layer `num_moe_layers`**
   (layer 40). There is no resident-NVFP4 MoE path (`layers/moe.py:60-92`,
   `engine/engine.py:1445-1453`), and the offload cache is fully layer-indexed —
   extending it by one layer is mechanical:
   - `_NVFP4_EXPERT_KEY_RE` (`models/qwen3_5_moe/weight.py:37-43`) gains an
     alternative that matches `mtp.layers.0.mlp.experts.*` → bank layer 40.
   - Widen `engine/engine.py:632` (`len(layers) == num_moe_layers`) and the
     bank-count checks (`models/nvfp4_banks.py:33-45`, `moe/offload_cache.py`)
     to `num_moe_layers + (1 if mtp else 0)`.
2. **MTP dense weights are ordinary model buffers** under `mtp.*` keys — new ops
   registered on the model so `load_state_dict` (strict, `layers/base.py:43,52`)
   balances. Loader change: `_rename` (`weight.py:136`) stops dropping `mtp.*`
   when MTP is enabled; Gemma +1 norm applies to the MTP layer's norms the same
   way (`weight.py:56-61`); NVFP4 dense triples for shared expert/attn per the
   existing emit paths.
3. **MTP attention gets global layer_id = 40** appended to the full-attention
   group → `MHAKVCache` auto-allocates an 11th slab (`kvcache/__init__.py:142-148`,
   `mha_pool.py:44-49`). KV for draft tokens is written normally and needs no
   erase-on-reject (page slots are per-request and overwritten).
4. **Verification reuses the FlashInfer "extend prefill" branch** —
   `fi.py:242-243` already computes ragged `cu_seqlens_q` for multi-token-per-seq
   causal attention over cached KV. Eager first; the captured decode wrapper
   stays 1-token. New batch mode: `phase="verify"` (`core.py:150-156` grows a
   third value) so GDN layers and the LM head branch correctly.
5. **GDN rollback via the kernel's built-in target_verify mode** —
   `fused_sigmoid_gating_recurrent.py` already supports
   `intermediate_states_buffer` + `cache_steps` + `disable_state_update`
   (`:208-244,263-324`). Verify runs the decode kernel with varlen cu_seqlens
   `[0,k,2k,...]`, caching per-step states; on acceptance of j ≤ k tokens, copy
   `buffer[slot, j-1]` into `recurrent_states[li, slot]` per layer.
   Conv state: save the (K-1)-wide window before verify (cheap:
   `conv_dim × (K-1) × 2B` per layer/seq), restore slice on partial accept.
6. **LM head all-logits mode**: the last-token slice (`layers/embedding.py:107-110`,
   `nvfp4_linear.py:925-927`) becomes conditional on the new phase — verify
   needs `[bs·(k+1), vocab]` logits.
7. **Draft/verify loop lives inside `Engine.forward_batch`** (seam 1;
   `engine.py:911-931`): target step → MTP drafts k tokens sequentially →
   verify step → accept/sample. `ForwardOutput` widens to `[bs, ≤k+1]` accepted
   tokens; `Req.complete_one()` generalizes to `complete_n(j)`; `_ids_buf`
   over-allocates by `draft_len` (`core.py:71-78,95`); `_make_write_tuple`
   (`scheduler.py:900-905`) writes j tokens.
8. **Gating**: `--mtp-drafts N` (default 0 = off), greedy-only in M2
   (`SamplingParams.is_greedy` gate), auto-disable at `batch.size > 2`
   (llama.cpp data shows MTP hurts at c≥4). NO confidence floor — measured
   counterproductive on GB10 (p-min 0.75: 112→81 tok/s).
9. **FTW**: existing FTW checkpoints lack mtp banks → conversion re-run needed
   when MTP enabled; detect missing `mtp` bank and fall back to safetensors
   load with a clear log line.

## Milestones

- **M1 — weights + offline validation.** Load MTP module (dense resident,
  experts via bank 40); standalone harness runs target prefill + MTP drafts on
  fixed prompts and reports top-1 agreement vs the target's own next tokens
  (expected ≈ published acceptance ~90%+). No scheduler changes.
- **M2 — greedy MTP-1 serve path, eager.** Draft 1 token, verify with the
  1-token decode path itself (verify of 1 draft ≡ decode step of 2 tokens);
  measure end-to-end tok/s.
- **M3 — multi-draft (k=2..4) verification** with GDN intermediate-state
  rollback + conv window restore; still eager. Target ≥95 tok/s.
- **M4 — CUDA-graph capture of draft+verify** ((bs, k)-keyed graphs, captured
  FI prefill wrapper), overlap-scheduling interaction, flags/docs/tests.

## Cross-cutting gotchas (from the code map)

- `Req._ids_buf` exact sizing + `append_host` assert (`core.py:71-78,95`).
- `complete_one()` called before sampling (`engine.py:923-924`) — ordering moves.
- Strict state_dict both directions (`layers/base.py:43,52-53`).
- `rotary max_position` table bound (`engine.py:1480-1492`) — draft positions
  must stay within budget.
- `[K,V]` vs `[V,K]` transpose coincidence at head_k==head_v==128
  (`gdn.py:60-65`) — intermediate-state copies must follow `_write_track_snapshot`.
- `ChunkedReq` is the precedent for Req subclasses with altered step semantics.

## M4 findings (2026-08-24) — kernel choice before graphs

Profiling an eager verify step (bs=1, k=1) reordered the plan:

- Eager 1-token decode costs 18.8 ms vs 16 ms graphed: launch overhead is ~3 ms,
  **not** the 13 ms gap to the verify step. The gap was kernel *choice*: the
  `verify` phase originally rode the prefill-path kernels, which are the wrong
  tool for 2 tokens — `ChunkGatedDeltaRuleFunction` cost ~8 ms of host time
  across 30 GDN layers for 1.4 ms of GPU work, and FlashInfer's prefill FMHA
  ~4.6 ms for 10 layers of 2-query attention.
- **GDN**: verify now uses the fused FLA *decode* kernel (varlen, T steps per
  sequence in-kernel) plus the triton conv update in multi-token mode
  (`seqlen=T`). Caveat that cost a day: the fused kernel models only the
  token stride; feed it contiguous q/k/v (a bs=1 transpose+reshape yields a
  strided view). Step: 35 → 26 ms.
- **Attention**: verify is presented to FlashInfer as `T` single-query
  pseudo-requests per request sharing the page-table row with KV lengths
  cached+1..device_len (all T K/V rows are stored before the attention call, so
  causality within the step holds by construction) → the fast decode kernel and
  the same wrapper/plan machinery as a decode batch. This also makes the verify
  step graph-capturable through the existing decode capture path
  (`FICaptureData` + `CUDAGraphBatchDecodeWithPagedKVCacheWrapper`) once the
  capture buffers are sized for `bs*T` rows and `fla_cu_seqlens = arange(0,
  bs*T+1, T)`.
- Remaining: `aten::_scaled_mm` (FP8 W8A8 dense projections) at M=2 — 6.5 ms;
  then CUDA-graph capture of the verify step keyed by (bs, T).

Debug knobs: `FREETOKEN_MTP_PROFILE=1` (per-step event breakdown + one-step
kernel table), `FREETOKEN_MTP_GDN_PATH=decode|mixed|chunk`,
`FREETOKEN_MTP_GDN_DUMP=<path>` (in-situ tensor dump of layer 0's first verify
step), `FREETOKEN_DISABLE_CUDA_GRAPH=1`.
