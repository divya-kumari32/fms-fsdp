"""Faithful Block Attention Residuals (arXiv 2603.15031, Fig. 2).

Separate from `ra_llama.py` (the strided i%N always-live variant, loss 2.604)
so both can be compared side by side. This module implements the paper's
contiguous-block scheme:

  * L layers -> N contiguous blocks of S = L/N layers; layer i -> block i//S.
  * Two depth reads per layer (before attn AND before MLP), each with its own
    zero-init pseudo-query + key-norm.
  * Two separate writes into a running `partial` (partial += attn_out, re-read,
    partial += mlp_out); the partial is FROZEN into a fixed slot at each block
    boundary and reset.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from fms.models.llama import LLaMAConfig
from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils.activation import str_to_activation


def _fix_scalar_params(module):
    """Replace scalar (0-d) parameters with 1D tensors for FSDP compatibility."""
    for name, param in list(module._parameters.items()):
        if param is not None and param.dim() == 0:
            module._parameters[name] = nn.Parameter(
                param.data.unsqueeze(0), requires_grad=param.requires_grad
            )
    for child in module.children():
        _fix_scalar_params(child)


def _depth_attend(query, keynorm, readset, mask):
    """Attention-residual read: softmax-weighted combination of the read set.

    Args:
        query:   [D]           — learned pseudo-query for this read (zero-init).
        keynorm: RMSNorm module applied to the read set before scoring.
        readset: [B, S, M, D]  — candidate values (M = N frozen blocks + 1 partial).
        mask:    [M]           — additive mask (0 for visible keys, -inf for
                                 future/empty blocks) so only completed blocks
                                 plus the active partial are attended.

    Returns:
        [B, S, D] — weighted combination of the visible read-set entries.
    """
    keys = keynorm(readset)
    scores = torch.einsum("d,bsmd->bsm", query, keys)
    scores = scores + mask.to(scores.dtype)
    attn_weights = F.softmax(scores, dim=-1)
    h = torch.einsum("bsm,bsmd->bsd", attn_weights, readset)
    return h


class BlockRATransformerBlock(nn.Module):
    """Pre-norm transformer block with per-sub-layer attention-residual reads.

    Each layer performs TWO depth reads — one before attention, one before the
    MLP — each with its own zero-init pseudo-query and key-norm (mirroring the
    paper's separate attn_res / mlp_res projections and norms). Both reads see
    the same visible set (completed frozen blocks + the active partial); only
    the partial's *value* changes between them (it has absorbed attn_out by the
    MLP read).

    The block reads `frozen_buf`/`partial` but does NOT freeze — the parent
    model handles block-boundary freezing. This keeps ONE FSDP all-gather per
    layer (all of attn/MLP/query/norm params live in this wrap unit and are
    gathered by a single forward call).

    FSDP wrap unit + AC checkpoint target.
    """

    def __init__(self, config: LLaMAConfig, rotary_emb):
        super().__init__()
        self.config = config
        emb_kq = config.emb_dim // config.nheads
        emb_v = config.emb_dim // config.nheads
        kvheads = config.kvheads if config.kvheads != 0 else config.nheads

        self.ln = LayerNormParameterized(
            config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=config.norm_eps,
            use_high_precision_pow=True,
        )
        self.attn = MultiHeadAttention(
            config.emb_dim,
            emb_kq,
            emb_v,
            config.nheads,
            kvheads,
            p_dropout=config.p_dropout,
            use_bias=config.attn_bias,
            position_encoder=rotary_emb,
            fused=config.fused_weights,
        )

        self.ff_ln = LayerNormParameterized(
            config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=config.norm_eps,
            use_high_precision_pow=True,
        )
        self.ff_sub_layer = GatedLinearUnit(
            config.emb_dim,
            hidden_grow_factor=config.hidden_grow_factor,
            multiple_of=config.multiple_of,
            activation_fn=str_to_activation(config.activation_fn),
            p_dropout=config.p_dropout,
            use_bias=config.mlp_bias,
            fused=config.fused_weights,
        )

        # Separate zero-init pseudo-query per read (→ uniform weights at start).
        self.attn_depth_query = nn.Parameter(torch.zeros(config.emb_dim))
        self.mlp_depth_query = nn.Parameter(torch.zeros(config.emb_dim))

        # Separate key-norm per read (paper uses distinct attn_res / mlp_res norms).
        self.attn_key_norm = LayerNormParameterized(
            config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=config.norm_eps,
            use_high_precision_pow=True,
        )
        self.mlp_key_norm = LayerNormParameterized(
            config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=config.norm_eps,
            use_high_precision_pow=True,
        )

    def forward(self, frozen_buf, partial, mask, *, position_ids=None, **kwargs):
        """One transformer layer with two attention-residual reads/writes.

        Args:
            frozen_buf: [B, S, N, D] — completed frozen block reps (zeros ahead).
            partial:    [B, S, D]    — running sum for the in-progress block.
            mask:       [N+1]        — visibility mask for this layer's block.

        Returns:
            partial:    [B, S, D]    — updated with attn_out then mlp_out.
        """
        # --- Attention sub-layer: read -> attn -> write into partial ---
        readset = torch.cat([frozen_buf, partial.unsqueeze(2)], dim=2)
        h = _depth_attend(self.attn_depth_query, self.attn_key_norm, readset, mask)
        attn_out = self.attn(
            q=self.ln(h), position_ids=position_ids, is_self=True, use_cache=False
        )
        partial = partial + attn_out

        # --- FFN sub-layer: re-read (sees attn_out) -> mlp -> write into partial ---
        readset = torch.cat([frozen_buf, partial.unsqueeze(2)], dim=2)
        h = _depth_attend(self.mlp_depth_query, self.mlp_key_norm, readset, mask)
        ffn_out = self.ff_sub_layer(self.ff_ln(h))
        partial = partial + ffn_out

        return partial


class BlockRALLaMA(nn.Module):
    """LLaMA with faithful Block Attention Residuals (arXiv 2603.15031, Fig. 2).

    The L layers are partitioned into N contiguous blocks of S = L / N layers.
    Layer i belongs to block b = i // S. Within a block, sub-layer outputs
    (attn_out, mlp_out) are accumulated into a running `partial`; when the
    block's S layers finish, `partial` is frozen into a fixed slot and a fresh
    partial is started. Each sub-layer READS via softmax attention over the set
    {embedding-seeded completed blocks, active partial} (Eq. 5:
    b_n = Σ_{j∈B_n} f_j(h_j)), keeping read memory O(N·d) regardless of depth.

    Differences from the earlier strided i%N variant (`ra_llama.RALLaMA`):
      * contiguous frozen blocks (i//S) instead of strided always-live slots (i%N);
      * two reads per layer (before attn AND before MLP), each with its own
        zero-init pseudo-query + key-norm;
      * two separate writes (partial += attn_out, re-read, partial += mlp_out).

    The final hidden state is Σ(frozen blocks) + partial, which equals
    embedding + Σ(all sub-layer deltas) — exactly the standard residual stream,
    since `partial` accumulates the raw deltas and freezing only relocates them.
    """

    def __init__(self, config: LLaMAConfig, *, cp_mesh=None, num_slots=4, ac_group_size=0):
        super().__init__()
        self.config = config
        # num_slots == number of blocks N; block_size == layers per block S.
        self.num_slots = num_slots
        assert config.nlayers % num_slots == 0, (
            f"nlayers ({config.nlayers}) must be divisible by num_slots/num_blocks "
            f"({num_slots}) for contiguous Block-AttnRes."
        )
        self.block_size = config.nlayers // num_slots
        # Number of transformer layers per activation-checkpoint region.
        # 0 = disabled: forward runs a plain layer loop (the per-block AC
        # handler in main_training applies checkpointing instead).
        self.ac_group_size = ac_group_size

        self.embedding = nn.Embedding(config.src_vocab_size, config.emb_dim)

        self.rot_emb = RotaryEmbedding(
            dim=config.emb_dim // config.nheads,
            ratio=config.rope_theta,
            max_seq_len=config.max_expected_seq_len,
        )
        self.rot_emb.compute_freqs_cis(
            torch.device("cpu"), config.max_expected_seq_len
        )

        self.layers = nn.ModuleList([
            BlockRATransformerBlock(config, self.rot_emb)
            for _ in range(config.nlayers)
        ])

        # NOTE: the [N, N+1] visibility mask is built per-forward in
        # `_visibility_mask` (a fresh activation tensor), NOT a registered
        # buffer. FSDP mixed precision casts registered buffers to the low
        # dtype via Tensor.set_(fp32_storage <- bf16), which torch.compile's
        # Dynamo cannot trace ("Could not set tensor of type BFloat16 to a
        # tensor of type float"). Building it in-graph sidesteps that entirely.

        self.dec_norm = LayerNormParameterized(
            config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=config.norm_eps,
            use_high_precision_pow=True,
        )

        self.head = nn.Linear(config.emb_dim, config.src_vocab_size, bias=False)
        self.embedding.weight = self.head.weight

        _fix_scalar_params(self)

    def reset_parameters(self):
        nn.init.trunc_normal_(
            self.embedding.weight, mean=0.0, std=self.config.emb_dim**-0.5
        )
        for device in set(
            [param.device for param in self.parameters()]
            + [buffer.device for buffer in self.buffers()]
        ):
            self.rot_emb.compute_freqs_cis(device, self.config.max_expected_seq_len)
        for m in self.modules():
            if isinstance(m, (MultiHeadAttention, GatedLinearUnit, LayerNormParameterized)):
                m.reset_parameters()
        # Pseudo-queries stay zero-init (uniform read weights at start).
        for layer in self.layers:
            nn.init.zeros_(layer.attn_depth_query)
            nn.init.zeros_(layer.mlp_depth_query)

    def _visibility_mask(self, device):
        """Additive read mask [N, N+1]; row b is the mask for block b.

        Built fresh each forward on the activation device (not a registered
        buffer) so FSDP mixed precision never casts it — casting the fp32 mask
        storage to bf16 via Tensor.set_ breaks torch.compile Dynamo tracing.

        Read-set order is [frozen_0 .. frozen_{N-1}, partial] (length N+1).
        Block b sees frozen blocks 0..b-1 (already frozen) and the partial
        (column N, always visible); columns b..N-1 are future/empty (-inf).
        """
        N = self.num_slots
        cols = torch.arange(N + 1, device=device)
        rows = torch.arange(N, device=device).unsqueeze(1)
        visible = (cols.unsqueeze(0) < rows) | (cols.unsqueeze(0) == N)
        mask = torch.zeros(N, N + 1, device=device)
        return mask.masked_fill(~visible, float("-inf"))

    def _run_layers(self, frozen_buf, partial, mask_table, position_ids, start, end):
        """Run layers [start, end) as one region.

        Carries (frozen_buf, partial) as the state between layers/groups:
          * frozen_buf [B,S,N,D] — completed block reps (zeros ahead of the
            current block); written only at block boundaries.
          * partial    [B,S,D]   — running sum for the in-progress block.
        Each layer reads over {frozen blocks, partial}, twice (attn + MLP), and
        the running partial absorbs both sub-layer deltas. At a block boundary
        the partial is frozen into its slot and reset. Returns (frozen_buf,
        partial) so the caller can reconstruct the residual stream at the end.
        """
        for i in range(start, end):
            b = i // self.block_size
            mask = mask_table[b]

            partial = self.layers[i](
                frozen_buf, partial, mask, position_ids=position_ids
            )

            # Freeze at the block boundary: relocate the completed partial into
            # its slot (slot is zeros -> add == copy, compile-safe functional
            # write) and start a fresh partial for the next block.
            if (i + 1) % self.block_size == 0:
                slots = list(frozen_buf.unbind(dim=2))
                slots[b] = slots[b] + partial
                frozen_buf = torch.stack(slots, dim=2)
                partial = torch.zeros_like(partial)

        return frozen_buf, partial

    def forward(self, x, position_ids=None, **kwargs):
        x = self.embedding(x)

        # frozen blocks start empty; the active partial is seeded with the
        # embedding, so block 0 carries the embedding (layer 0 then reads
        # exactly the embedding -> matches a standard transformer at init).
        frozen_buf = torch.zeros(
            x.shape[0], x.shape[1], self.num_slots, x.shape[2],
            dtype=x.dtype, device=x.device,
        )
        partial = x

        # Visibility mask built in-graph on the activation device (see
        # _visibility_mask); passed through so the checkpointed region sees it.
        mask_table = self._visibility_mask(x.device)

        if self.ac_group_size > 0 and self.training:
            # Grouped activation checkpointing: checkpoint each region of
            # `ac_group_size` layers. Fewer stored boundary snapshots than
            # per-block AC (one per group instead of one per layer), at the
            # cost of recomputing the group's reads/writes in backward.
            nlayers = len(self.layers)
            for start in range(0, nlayers, self.ac_group_size):
                end = min(start + self.ac_group_size, nlayers)
                frozen_buf, partial = checkpoint(
                    self._run_layers, frozen_buf, partial, mask_table,
                    position_ids, start, end,
                    use_reentrant=False,
                )
        else:
            # Disabled (group_size=0) or eval: plain loop. Per-block AC (if
            # enabled) is applied externally by the training script's AC handler.
            frozen_buf, partial = self._run_layers(
                frozen_buf, partial, mask_table, position_ids, 0, len(self.layers)
            )

        # Reconstruct the residual stream: Σ(frozen blocks) + active partial
        # = embedding + Σ(all sub-layer deltas).
        x = frozen_buf.sum(dim=2) + partial

        x = self.dec_norm(x)
        return self.head(x)
