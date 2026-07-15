import torch
import torch.nn as nn
import torch.nn.functional as F
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
    """LLaMA with Full Attention Residuals (arXiv 2603.15031).

    Replaces fixed unit-weight residual accumulation with softmax attention
    over preceding layer outputs. Each layer l computes:

        h_l = Σ_{i=0}^{l-1} α_{i→l} · v_i

    where v_0 = embedding, v_i = f_i(h_i) (layer i's contribution),
    α_{i→l} = softmax(q_l^T · RMSNorm(v_i)) with learned pseudo-query q_l.

    This is the full version (no compression): cache grows linearly per layer.
    """

    def __init__(self, config: LLaMAConfig, *, cp_mesh=None):
        super().__init__()
        self.config = config

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
        # Shape: [nlayers, emb_dim]
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
        # depth_queries already zero-initialized (uniform attention at start)

    def _depth_attend(self, query, values):
        """Compute attention residual: weighted sum of cached values.

        Args:
            query: [D] — learned pseudo-query for this layer
            values: [B, S, N, D] — stacked cache entries (N = num entries so far)

        Returns:
            [B, S, D] — weighted combination of cached values
        """
        # Apply RMSNorm to keys (= values, tied)
        keys = self.key_norm(values)  # [B, S, N, D]

        # Score: dot product between query and each normalized key
        # query [D] broadcast against keys [B, S, N, D] → scores [B, S, N]
        scores = torch.einsum("d,bsnd->bsn", query, keys)

        # Softmax over cache entries (depth-wise attention)
        attn_weights = F.softmax(scores, dim=-1)  # [B, S, N]

        # Weighted sum of raw values (not normalized)
        h = torch.einsum("bsn,bsnd->bsd", attn_weights, values)  # [B, S, D]
        return h

    def forward(self, x, position_ids=None, **kwargs):
        x = self.embedding(x)

        # v_0 = token embedding (first cache entry)
        cache_entries = [x]

        for i, layer in enumerate(self.layers):
            # h_l = Σ α_{i→l} · v_i  (attention over all cached entries)
            cache_tensor = torch.stack(cache_entries, dim=2)  # [B, S, i+1, D]
            x = self._depth_attend(self.depth_queries[i], cache_tensor)

            # Run transformer block, get both output and raw contribution
            x, contribution = layer(x, position_ids=position_ids)

            # Cache the layer's contribution: v_i = f_i(h_i)
            cache_entries.append(contribution)

        x = self.dec_norm(x)
        return self.head(x)
