from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from freetoken.env import ENV
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    # Tokens per request the buffer was built for (1 = decode; T = MTP verify steps).
    tokens_per_req: int = 1

    @classmethod
    def init(
        cls, bs: int, vocab_size: int, device: torch.device, tokens_per_req: int = 1
    ) -> GraphCaptureBuffer:
        rows = bs * tokens_per_req
        return GraphCaptureBuffer(
            input_ids=torch.zeros(rows, dtype=torch.int32, device=device),
            out_loc=torch.zeros(rows, dtype=torch.int32, device=device),
            positions=torch.zeros(rows, dtype=torch.int32, device=device),
            logits=torch.empty(rows, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            # GDN query indptr: arange(0, rows+1, T) -- a constant per captured bs.
            fla_cu_seqlens=torch.arange(
                0, rows + 1, tokens_per_req, dtype=torch.int32, device=device
            ),
            tokens_per_req=tokens_per_req,
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata

        bs = batch.padded_size
        rows = bs * self.tokens_per_req
        batch.input_ids = self.input_ids[:rows]
        batch.out_loc = self.out_loc[:rows]
        batch.positions = self.positions[:rows]
        batch.linear_table_idx = self.table_idx[:bs]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            cu_seqlens=self.fla_cu_seqlens[: bs + 1], cache_indices=self.table_idx[:bs]
        )

    def copy_from(self, batch: Batch) -> None:
        bs = batch.padded_size
        rows = bs * self.tokens_per_req
        self.input_ids[:rows] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[:rows] = batch.out_loc
        self.positions[:rows] = batch.positions
        if batch.linear_table_idx is not None:
            self.table_idx[:bs] = batch.linear_table_idx


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
        verify_tokens: int = 0,
        verify_post=None,
    ) -> None:
        # MTP verify steps (T tokens per request) get their own graph family, keyed
        # ("verify", bs). 0 = no verify graphs. verify_post(batch, logits) -> tensors is
        # captured right after the forward so replay yields the sampled/accept/draft
        # results without any eager work.
        self.verify_tokens = verify_tokens
        self.verify_post = verify_post
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.stream = stream
        self.device = device
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, phase="decode")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        self._reset_moe_offload_cache()
        if self.verify_tokens > 1:
            self._capture_verify_graphs(vocab_size, model, pool)
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _capture_verify_graphs(self, vocab_size: int, model: BaseLLMModel, pool) -> None:
        """MTP verify graphs: T tokens per request. The attention backend sees a verify
        batch as bs*T single-query pseudo-requests, so it needs a decode graph wrapper for
        bs*T rows -- only batch sizes whose bs*T is itself a captured decode size get a
        verify graph (bs in {1,2,4} at T=2 -> rows {2,4,8})."""
        import copy

        T = self.verify_tokens
        self.verify_bs_list = sorted(
            bs for bs in self.graph_bs_list if bs * T in self.attn_backend.capture_bs
        )
        if not self.verify_bs_list:
            return logger.info_rank0("No verify CUDA graphs: no matching decode sizes.")
        self.verify_buffer = GraphCaptureBuffer.init(
            max(self.verify_bs_list), vocab_size, self.device, tokens_per_req=T
        )
        self.verify_hidden: Dict[int, torch.Tensor | None] = {}
        self.verify_outs: Dict[int, tuple | None] = {}
        self._model = model
        # A verify dummy: same slots as the decode dummy, extend_len == T.
        vdummy = copy.copy(self.dummy_req)
        vdummy.device_len = vdummy.cached_len + T
        self.verify_dummy_req = vdummy
        dummy_slot = (vdummy.linear_slot_idx if vdummy.linear_slot_idx is not None
                      else vdummy.table_idx)
        logger.info_rank0(f"Capturing MTP verify graphs (T={T}) for bs {self.verify_bs_list}")
        for bs in sorted(self.verify_bs_list, reverse=True):
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[vdummy] * bs, phase="verify")
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.verify_buffer.set_batch(batch)
            self.verify_buffer.table_idx[:bs].fill_(dummy_slot)
            rows = bs * T
            with get_global_ctx().forward_batch(batch):
                self.verify_buffer.logits[:rows] = model.forward()
                if self.verify_post is not None:
                    self.verify_post(batch, self.verify_buffer.logits[:rows])
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.verify_buffer.logits[:rows] = model.forward()
                    outs = (
                        self.verify_post(batch, self.verify_buffer.logits[:rows])
                        if self.verify_post is not None else None
                    )
                self._reset_moe_offload_cache()
            self.graph_map[("verify", bs)] = graph
            # Graph-owned outputs (stable addresses across replays): the post-processing
            # tensors, and model._mtp_hidden which model.forward() stashes during capture
            # but which a replay never re-stashes.
            self.verify_outs[bs] = outs
            self.verify_hidden[bs] = getattr(model, "_mtp_hidden", None)
        self._reset_moe_offload_cache()

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        if ENV.DISABLE_CUDA_GRAPH:  # profiling knob: force eager
            return False
        if getattr(batch, "is_verify", False):
            return (
                self.verify_tokens > 1
                and bool(getattr(self, "verify_bs_list", None))
                and batch.size <= max(self.verify_bs_list)
                and all(r.extend_len == self.verify_tokens for r in batch.reqs)
            )
        return batch.is_decode and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        if getattr(batch, "is_verify", False):
            self.verify_buffer.copy_from(batch)
            bs = batch.padded_size
            g = self.graph_map[("verify", bs)]
            self.attn_backend.prepare_for_replay(batch)
            g.replay()
            self._model._mtp_hidden = self.verify_hidden[bs]
            batch._verify_graph_out = self.verify_outs[bs]
            return self.verify_buffer.logits[: batch.size * self.verify_tokens]
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def pad_batch(self, batch: Batch) -> None:
        if not self.can_use_cuda_graph(batch):
            batch.padded_reqs = batch.reqs
            return
        if getattr(batch, "is_verify", False):
            padded_size = next(bs for bs in self.verify_bs_list if bs >= batch.size)
            batch.padded_reqs = batch.reqs + [self.verify_dummy_req] * (padded_size - batch.size)
            return
        padded_size = next(bs for bs in self.graph_bs_list if bs >= batch.size)
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.buffer = None
        gc.collect()
