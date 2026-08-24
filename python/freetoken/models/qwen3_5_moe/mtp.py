"""Multi-Token-Prediction (MTP) draft head for Qwen3.5/3.6 MoE checkpoints.

The checkpoint ships a DeepSeek-style MTP module under ``mtp.*``:

    h' = fc(concat(pre_fc_norm_embedding(embed(tok)), pre_fc_norm_hidden(hidden)))
    h' -> one full-attention decoder layer (gated attention + routed MoE + shared expert)
    logits = lm_head(norm(h'))            # shared lm_head / embeddings

Unlike the target model's layers, the whole head is stored BF16 (including its
256 routed experts, ~1.6 GiB), so the experts live GPU-resident as a plain
``MoELayer(weight_format="bf16")`` — no offload bank involvement. The dense
projections reuse the model's own op classes through ``_BF16View``, a config
proxy that forces the bf16 dispatch in ``make_col_merged``/``make_replicated``
and ``_SharedExpert`` regardless of the target model's quantization.

Enablement: ``FREETOKEN_MTP=1`` (milestone 1; a proper ``--mtp-drafts`` engine
flag replaces the env var when the serve path lands).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from freetoken.layers import BaseOP, GemmaRMSNorm, LinearReplicated, OPList
from freetoken.layers.moe import MoELayer
from freetoken.moe.fused import fused_topk
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .moe import _SharedExpert

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


def mtp_enabled() -> bool:
    return os.environ.get("FREETOKEN_MTP", "0") == "1"


class _BF16View:
    """Config proxy forcing the bf16 dense-linear dispatch for the MTP head's ops.

    Class attributes shadow ``__getattr__`` (which only fires on misses), so the
    quant fields read "none" while everything else delegates to the real config.
    """

    expert_quant = "none"
    attn_quant = "none"
    dense_quant = "none"
    lm_head_quant = "none"

    def __init__(self, config: "ModelConfig"):
        self._config = config

    def __getattr__(self, name: str):
        return getattr(self._config, name)


class _MTPMoE(BaseOP):
    """Routed MoE + gated shared expert of the MTP layer, all BF16-resident.

    Mirrors ``Qwen3_5MoE`` (same state-dict keys) but routes explicitly through
    ``fused_topk`` + ``MoELayer.routed_forward`` instead of ``ctx.moe_backend``,
    which in an offload-configured engine belongs to the offloaded target layers.
    """

    def __init__(self, config: "ModelConfig"):
        self.experts = MoELayer(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=True,
            weight_format="bf16",
        )
        view = _BF16View(config)
        self.gate = LinearReplicated(config.hidden_size, config.num_experts, has_bias=False)
        self.shared_expert = _SharedExpert(
            view, config.hidden_size, config.shared_expert_intermediate_size
        )
        self.shared_expert_gate = LinearReplicated(config.hidden_size, 1, has_bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate.forward(hidden_states)
        shared = self.shared_expert.forward(hidden_states)
        shared = shared * torch.sigmoid(self.shared_expert_gate.forward(hidden_states))
        topk_weights, topk_ids = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.experts.top_k,
            renormalize=True,
        )
        # routed_forward's contract allows in-place topk_ids mutation; pass a clone.
        routed = self.experts.routed_forward(hidden_states, topk_weights, topk_ids.clone())
        return (routed + shared).view(num_tokens, hidden_dim)


class _MTPDecoderLayer(BaseOP):
    """Pre-norm full-attention block of the MTP head (state-dict-compatible with
    ``mtp.layers.N.*``): same residual-stream form as ``Qwen3_5DecoderLayer``."""

    def __init__(self, config: "ModelConfig", layer_id: int):
        self._layer_id = layer_id
        self.self_attn = Qwen3_5Attention(_BF16View(config), layer_id)
        self.mlp = _MTPMoE(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("MTPLayer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5MTPHead(BaseOP):
    """The draft head. ``forward`` maps (token embeddings, target hidden states)
    -> normalized hidden states ready for the shared lm_head.

    The MTP layer's attention runs against the engine's paged KV via
    ``ctx.attn_backend`` with global layer_id ``num_layers + i`` — wiring of the
    extra KV slab and batch metadata is the serve-path milestone; construction
    and weight loading are complete here.
    """

    def __init__(self, config: "ModelConfig"):
        h, eps = config.hidden_size, config.rms_norm_eps
        self.pre_fc_norm_embedding = GemmaRMSNorm(h, eps=eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(h, eps=eps)
        self.fc = LinearReplicated(2 * h, h, has_bias=False)
        self.layers = OPList(
            [
                _MTPDecoderLayer(config, config.num_layers + i)
                for i in range(config.mtp_num_layers)
            ]
        )
        self.norm = GemmaRMSNorm(h, eps=eps)

    def combine(self, token_embeds: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """The pre-layer combiner. Order matches the checkpoint naming
        (embedding half first); if measured draft acceptance comes out near zero
        with correct weights, swapping the halves is the first thing to try."""
        x = torch.cat(
            [
                self.pre_fc_norm_embedding.forward(token_embeds),
                self.pre_fc_norm_hidden.forward(hidden),
            ],
            dim=-1,
        )
        return self.fc.forward(x)

    def forward(self, token_embeds: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        x = self.combine(token_embeds, hidden)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, _ = self.norm.forward_add_residual(x, residual)
        return x


__all__ = ["Qwen3_5MTPHead", "mtp_enabled"]
