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


class RATransformerBlock(nn.Module):
    """Pre-norm transformer block that returns BOTH the residual output and
    the raw layer contribution (for the attention residuals cache).

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

    def forward(self, x, *, position_ids=None, **kwargs):
        # Attention sub-layer
        attn_out = self.attn(
            q=self.ln(x), position_ids=position_ids, is_self=True, use_cache=False
        )
        x = x + attn_out

        # FFN sub-layer
        ffn_out = self.ff_sub_layer(self.ff_ln(x))
        x = x + ffn_out

        # Return accumulated state AND the raw contribution (attn + ffn)
        # The contribution is what gets cached for attention residuals
        contribution = attn_out + ffn_out
        return x, contribution


class RALLaMA(nn.Module):
    """LLaMA with Bounded Attention Residuals (arXiv 2603.15031, i%n variant).

    Replaces fixed unit-weight residual accumulation with softmax attention
    over a fixed-size cache of N slots. Each layer l computes:

        h_l = Σ_{i=0}^{N-1} α_{i→l} · cache[i]

    Layer contributions are ACCUMULATED into slot (layer_idx % N) via residual
    add (slot += contribution), so each slot is a persistent residual stream
    (embedding + the deltas routed to it) rather than a bare overwritten delta.
    Keeps memory constant regardless of depth.
    """

    def __init__(self, config: LLaMAConfig, *, cp_mesh=None, num_slots=4, ac_group_size=0):
        super().__init__()
        self.config = config
        self.num_slots = num_slots
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
            RATransformerBlock(config, self.rot_emb)
            for _ in range(config.nlayers)
        ])

        # Learned pseudo-query per layer (zero-init → uniform weights at start)
        self.depth_queries = nn.Parameter(torch.zeros(config.nlayers, config.emb_dim))

        # RMSNorm for keys (applied to cached values before scoring)
        self.key_norm = LayerNormParameterized(
            config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=config.norm_eps,
            use_high_precision_pow=True,
        )

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

    def _depth_attend(self, query, cache):
        """Compute attention residual: weighted sum of cached values.

        Args:
            query: [D] — learned pseudo-query for this layer
            cache: [B, S, N, D] — fixed-size cache (N = num_slots)

        Returns:
            [B, S, D] — weighted combination of cache slots
        """
        keys = self.key_norm(cache)
        scores = torch.einsum("d,bsnd->bsn", query, keys)
        attn_weights = F.softmax(scores, dim=-1)
        h = torch.einsum("bsn,bsnd->bsd", attn_weights, cache)
        return h

    def _run_layers(self, cache, position_ids, start, end):
        """Run layers [start, end) as one region: read → block → write per layer.

        Carries only `cache` ([B,S,N,D]) as state between layers — the read
        output `x` is fully re-derived from `cache` at the top of each layer,
        so `cache` is the single true carried tensor. Returns (x, cache) where
        x is the last layer's block output (needed for the final dec_norm/head).
        """
        x = None
        for i in range(start, end):
            # Read: attend over fixed-size cache
            x = self._depth_attend(self.depth_queries[i], cache)

            # Transform: run through transformer block
            x, contribution = self.layers[i](x, position_ids=position_ids)

            # Write: ACCUMULATE contribution into slot i % N (residual add, not
            # overwrite). Adding preserves the identity/residual path through the
            # slot — overwriting with the bare delta destroys it and makes the
            # gradient devolve into noise. Each slot is a residual accumulator
            # (embedding + the deltas routed to it), matching attention-residuals'
            # compressed variant. Use unbind/stack for compile-safe functional write.
            slot = i % self.num_slots
            slots = list(cache.unbind(dim=2))
            slots[slot] = slots[slot] + contribution
            cache = torch.stack(slots, dim=2)

        return x, cache

    def forward(self, x, position_ids=None, **kwargs):
        x = self.embedding(x)

        # Initialize all N cache slots from embedding
        cache = x.unsqueeze(2).expand(-1, -1, self.num_slots, -1).clone()

        if self.ac_group_size > 0 and self.training:
            # Grouped activation checkpointing: checkpoint each region of
            # `ac_group_size` layers. Fewer stored boundary snapshots than
            # per-block AC (one per group instead of one per layer), at the
            # cost of recomputing the group's read→block→write in backward.
            nlayers = len(self.layers)
            for start in range(0, nlayers, self.ac_group_size):
                end = min(start + self.ac_group_size, nlayers)
                x, cache = checkpoint(
                    self._run_layers, cache, position_ids, start, end,
                    use_reentrant=False,
                )
        else:
            # Disabled (group_size=0) or eval: plain loop, byte-equivalent to
            # the original per-layer forward. Per-block AC (if enabled) is
            # applied externally by the training script's AC handler.
            x, cache = self._run_layers(cache, position_ids, 0, len(self.layers))

        x = self.dec_norm(x)
        return self.head(x)
