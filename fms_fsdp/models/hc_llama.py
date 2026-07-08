import torch
import torch.nn as nn
from fms.models.llama import LLaMAConfig
from fms.modules.attention import MultiHeadAttention
from fms.modules.feedforward import GatedLinearUnit
from fms.modules.layernorm import LayerNormParameterized
from fms.modules.positions import RotaryEmbedding
from fms.utils.activation import str_to_activation
from hyper_connections import mc_get_init_and_expand_reduce_stream_functions


def _fix_scalar_params(module):
    """Replace scalar (0-d) parameters with 1D tensors for FSDP compatibility."""
    for name, param in list(module._parameters.items()):
        if param is not None and param.dim() == 0:
            module._parameters[name] = nn.Parameter(
                param.data.unsqueeze(0), requires_grad=param.requires_grad
            )
    for child in module.children():
        _fix_scalar_params(child)


class AttnBranch(nn.Module):
    def __init__(self, ln, attn):
        super().__init__()
        self.ln = ln
        self.attn = attn

    def forward(self, x, **kwargs):
        return self.attn(
            q=self.ln(x),
            position_ids=kwargs.get("position_ids"),
            is_self=True,
            use_cache=False,
        )


class FFNBranch(nn.Module):
    def __init__(self, ln, ffn):
        super().__init__()
        self.ln = ln
        self.ffn = ffn

    def forward(self, x, **kwargs):
        return self.ffn(self.ln(x))


class HCLLaMABlock(nn.Module):
    def __init__(self, config: LLaMAConfig, rotary_emb, init_hc, layer_idx):
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

        self.hc_attn = init_hc(
            dim=config.emb_dim,
            branch=AttnBranch(self.ln, self.attn),
            layer_index=layer_idx * 2,
        )
        self.hc_ffn = init_hc(
            dim=config.emb_dim,
            branch=FFNBranch(self.ff_ln, self.ff_sub_layer),
            layer_index=layer_idx * 2 + 1,
        )

    def forward(self, x, *, position_ids=None, **kwargs):
        x = self.hc_attn(x, position_ids=position_ids)
        if torch.isnan(x).any():
            print(f"[NaN] in block after hc_attn", flush=True)
            return x
        x = self.hc_ffn(x)
        if torch.isnan(x).any():
            print(f"[NaN] in block after hc_ffn", flush=True)
        return x


class HCLLaMA(nn.Module):
    def __init__(self, config: LLaMAConfig, *, num_streams=4, sinkhorn_iters=20, cp_mesh=None):
        super().__init__()
        self.config = config
        self.num_streams = num_streams

        init_hc, self.expand_stream, self.reduce_stream = (
            mc_get_init_and_expand_reduce_stream_functions(num_streams)
        )

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
            HCLLaMABlock(config, self.rot_emb, init_hc, layer_idx=i)
            for i in range(config.nlayers)
        ])

        self.dec_norm = LayerNormParameterized(
            config.emb_dim,
            elementwise_scale=True,
            elementwise_shift=False,
            use_mean=False,
            eps=config.norm_eps,
            use_high_precision_pow=True,
        )

        self.head = nn.Linear(config.emb_dim, config.src_vocab_size, bias=False)
        # Weight tying
        self.embedding.weight = self.head.weight

        _fix_scalar_params(self)

    def forward(self, x, position_ids=None, **kwargs):
        x = self.embedding(x)
        x = self.expand_stream(x)

        if position_ids is not None:
            position_ids = position_ids.repeat(self.num_streams, 1)

        for i, layer in enumerate(self.layers):
            x = layer(x, position_ids=position_ids)
            if torch.isnan(x).any():
                print(f"[NaN] detected after layer {i}", flush=True)
                break

        x = self.reduce_stream(x)
        x = self.dec_norm(x)
        return self.head(x)
