#!/usr/bin/env python3
"""Milestone-1 check for the Qwen3.5/3.6 MTP draft head.

Builds ``Qwen3_5MTPHead`` from the checkpoint config, streams the ``mtp.*``
tensors through the production weight loader (fusions, merges, Gemma +1), does
a STRICT load_state_dict in both directions, and smoke-runs every ctx-free
compute path (combiner, router, shared expert, resident bf16 expert GEMM).
The attention sublayer needs the engine's paged-KV context and is exercised by
the serve-path milestone instead.

Usage: FREETOKEN_MTP=1 python scripts/mtp_m1_check.py <model_path>
"""

import os
import sys

os.environ.setdefault("FREETOKEN_MTP", "1")

import torch  # noqa: E402


def main() -> int:
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/models/Qwen3.6-35B-A3B-NVFP4"
    device = torch.device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device(device)

    from freetoken.distributed.info import set_tp_info
    from freetoken.utils.hf import cached_load_hf_config

    set_tp_info(rank=0, size=1)
    from freetoken.models.qwen3_5_moe.config import parse_config
    from freetoken.models.qwen3_5_moe.mtp import Qwen3_5MTPHead
    from freetoken.models.qwen3_5_moe.weight import iter_weights

    config = parse_config(cached_load_hf_config(model_path))
    assert config.mtp_num_layers > 0, "checkpoint has no MTP head"
    print(f"config: mtp_num_layers={config.mtp_num_layers}, hidden={config.hidden_size}, "
          f"experts={config.num_experts} top{config.num_experts_per_tok} "
          f"I={config.moe_intermediate_size}")

    head = Qwen3_5MTPHead(config)
    expected = head.state_dict(prefix="mtp")
    print(f"model buffers under mtp.*: {len(expected)}")

    mtp_sd = {}
    for name, tensor in iter_weights(
        model_path, device, include_moe_experts=False, include_non_moe=True
    ):
        if name.startswith("mtp."):
            mtp_sd[name] = tensor
    print(f"loader yielded mtp tensors: {len(mtp_sd)}")

    missing = sorted(set(expected) - set(mtp_sd))
    unexpected = sorted(set(mtp_sd) - set(expected))
    assert not missing, f"missing from checkpoint stream: {missing}"
    assert not unexpected, f"unexpected from checkpoint stream: {unexpected}"
    for k in expected:
        want, got = expected[k], mtp_sd[k]
        assert want.shape == got.shape, f"{k}: shape {got.shape} != buffer {want.shape}"
        # materialize the way the engine does: cast to the buffer's declared dtype
        mtp_sd[k] = got.to(want.dtype)

    head.load_state_dict(dict(mtp_sd), prefix="mtp")  # strict: raises on leftovers
    print("STRICT LOAD OK — names, shapes, dtypes all balanced")

    # --- ctx-free numeric smoke ---------------------------------------------
    torch.manual_seed(0)
    n = 4
    token_embeds = torch.randn(n, config.hidden_size) * 0.02
    hidden = torch.randn(n, config.hidden_size)

    x = head.combine(token_embeds, hidden)
    assert x.shape == (n, config.hidden_size) and torch.isfinite(x.float()).all()
    print(f"combiner ok: out norm={x.float().norm():.3f}")

    layer = head.layers.op_list[0]
    normed = layer.input_layernorm.forward(hidden)
    mlp_out = layer.mlp.forward(normed)
    assert mlp_out.shape == (n, config.hidden_size) and torch.isfinite(mlp_out.float()).all()
    print(f"MoE mlp ok (router + shared expert + resident bf16 expert GEMM): "
          f"out norm={mlp_out.float().norm():.3f}")

    gpu_bytes = sum(t.numel() * t.element_size() for t in head.state_dict(prefix="mtp").values())
    print(f"MTP head resident size: {gpu_bytes / (1 << 30):.2f} GiB")
    print("M1 CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
