"""Talkie 13B transformer — patched for long-context SFT.

Differences vs lewtun/talkie-1930-13b-it-hf upstream:
1. Liger fused linear cross-entropy in the loss path so the float32 logits
   tensor (shape S x V) is never materialised in HBM. Roughly 16 GB saved at
   S=64K, V=65540.
2. FlashAttention varlen path keyed off `position_ids`. When TRL passes a
   packed sequence (padding_free=True), tokens from different documents do
   not attend across boundaries.
3. Gradient checkpointing on the decoder stack.
4. RoPE precompute is configurable via config.max_position_embeddings; we set
   it to 64K at load time.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .configuration_talkie import TalkieConfig

try:
    from flash_attn import flash_attn_varlen_func
    _HAS_FA = True
except ImportError:
    _HAS_FA = False

try:
    from liger_kernel.transformers.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyLoss,
    )
    _HAS_LIGER = True
except ImportError:
    _HAS_LIGER = False


from dataclasses import dataclass, field


@dataclass
class TalkieCausalLMOutput(CausalLMOutputWithPast):
    """CausalLMOutputWithPast plus a token_accuracy field expected by TRL when
    SFTConfig.use_liger_kernel=True."""
    token_accuracy: Optional[torch.Tensor] = None


class TalkieHeadGain(nn.Module):
    def __init__(self, n_head: int):
        super().__init__()
        self.head_g = nn.Parameter(torch.ones(n_head))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.head_g.type_as(x).view(1, 1, -1, 1)


class TalkieWeightGain(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_g = nn.Parameter(torch.ones(1))

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        return w * self.w_g.type_as(w)


class TalkieActGain(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.a_g = nn.Parameter(torch.ones(1) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.a_g.type_as(x)


def _apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3).type_as(x)


def _precompute_rotary_embeddings(
    seq_len: int, head_dim: int, base: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos.bfloat16(), sin.bfloat16()
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return cos, sin


def _gather_rope_per_position(
    cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Index RoPE tables by position_ids.

    cos/sin: (1, S_table, 1, D_half)
    position_ids: (B, S)
    returns (B, S, 1, D_half) bf16
    """
    cos_t = cos[0, :, 0, :]  # (S_table, D_half)
    sin_t = sin[0, :, 0, :]
    flat = position_ids.reshape(-1)
    cos_g = cos_t.index_select(0, flat).reshape(*position_ids.shape, 1, cos_t.shape[-1])
    sin_g = sin_t.index_select(0, flat).reshape(*position_ids.shape, 1, sin_t.shape[-1])
    return cos_g, sin_g


def _cu_seqlens_from_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    """Convert per-token position_ids (where each new doc restarts at 0) into
    cu_seqlens suitable for flash_attn_varlen_func.

    Expects shape (B, S). For B>1 flatten before calling. Returns only cu_seqlens;
    the caller can pass the total sequence length as an over-approximation of
    max_seqlen to avoid a forced .item() sync (which torch.compile breaks on).
    """
    pos = position_ids.reshape(-1)
    starts = (pos == 0).nonzero(as_tuple=False).squeeze(-1)
    cu = torch.cat(
        [starts, torch.tensor([pos.numel()], device=pos.device, dtype=starts.dtype)]
    ).to(torch.int32)
    return cu


class TalkieSelfAttention(nn.Module):
    is_causal = True

    def __init__(self, config: TalkieConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_head = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        n_state = config.hidden_size

        self.attn_query = nn.Linear(n_state, n_state, bias=False)
        self.attn_key = nn.Linear(n_state, n_state, bias=False)
        self.attn_value = nn.Linear(n_state, n_state, bias=False)
        self.attn_resid = nn.Linear(n_state, n_state, bias=False)
        self.head_gain = TalkieHeadGain(config.num_attention_heads)

    def forward(
        self,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.size()
        q = self.attn_query(x).view(bsz, seq_len, self.n_head, self.head_dim)
        k = self.attn_key(x).view(bsz, seq_len, self.n_head, self.head_dim)
        v = self.attn_value(x).view(bsz, seq_len, self.n_head, self.head_dim)

        cos, sin = cos_sin
        q, k = _apply_rotary_emb(q, cos, sin), _apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q = self.head_gain(q)

        if cu_seqlens is not None and _HAS_FA:
            assert bsz == 1, "varlen path expects flattened batch"
            q_f = q.reshape(seq_len, self.n_head, self.head_dim)
            k_f = k.reshape(seq_len, self.n_head, self.head_dim)
            v_f = v.reshape(seq_len, self.n_head, self.head_dim)
            y = flash_attn_varlen_func(
                q_f,
                k_f,
                v_f,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
            )
            y = y.reshape(bsz, seq_len, self.n_head * self.head_dim)
        else:
            attn_impl = getattr(self.config, "_attn_implementation", "sdpa")
            attn_fn = ALL_ATTENTION_FUNCTIONS.get(attn_impl)
            if attn_fn is None:
                attn_fn = ALL_ATTENTION_FUNCTIONS["sdpa"]
            y, _ = attn_fn(
                self,
                q.transpose(1, 2),
                k.transpose(1, 2),
                v.transpose(1, 2),
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0,
                is_causal=True,
                **kwargs,
            )
            y = y.contiguous().view(bsz, seq_len, self.n_head * self.head_dim)
        return self.attn_resid(y)


class TalkieMLP(nn.Module):
    def __init__(self, config: TalkieConfig):
        super().__init__()
        n_state = config.hidden_size
        n_mlp = config.intermediate_size

        self.mlp_gate = nn.Linear(n_state, n_mlp, bias=False)
        self.mlp_linear = nn.Linear(n_state, n_mlp, bias=False)
        self.mlp_resid = nn.Linear(n_mlp, n_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_resid(F.silu(self.mlp_gate(x)) * self.mlp_linear(x))


class TalkieDecoderLayer(nn.Module):
    def __init__(self, config: TalkieConfig, layer_idx: int = 0):
        super().__init__()
        gain_init = (2 * config.num_hidden_layers) ** -0.5

        self.layer_idx = layer_idx
        self.attn = TalkieSelfAttention(config, layer_idx=layer_idx)
        self.attn_gain = TalkieActGain(gain_init)
        self.mlp = TalkieMLP(config)
        self.mlp_gain = TalkieActGain(gain_init)
        self.embed_skip = TalkieActGain(0.0)

    def forward(
        self,
        e_x: torch.Tensor,
        x: torch.Tensor,
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        x = x + self.attn_gain(
            self.attn(
                F.rms_norm(x, (x.shape[-1],)),
                cos_sin,
                cu_seqlens,
                max_seqlen,
                **kwargs,
            )
        )
        x = x + self.mlp_gain(self.mlp(F.rms_norm(x, (x.shape[-1],))))
        x = x + self.embed_skip(e_x)
        return x


class TalkieModel(PreTrainedModel):
    """Decoder stack — HF-style forward so vLLM's transformers backend
    (`AutoModel.from_config(...)`) can host this model."""

    config_class = TalkieConfig
    _no_split_modules = ["TalkieDecoderLayer"]
    _supports_gradient_checkpointing = True
    _supports_attention_backend = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True
    base_model_prefix = "model"
    # Empty plan = single-GPU / replicate. Multi-GPU TP would need entries
    # for q/k/v/o-proj. vLLM tolerates an empty plan when world_size==1.
    tp_plan = {}

    def __init__(self, config: TalkieConfig):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList(
            [
                TalkieDecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.gradient_checkpointing = False
        # Selective activation checkpointing: only checkpoint every Nth layer.
        # stride=1 => every layer (HF default), stride=2 => half of layers,
        # stride=N => no layers checkpointed. Set via env at construction time.
        import os as _os
        try:
            self.gc_stride = max(1, int(_os.environ.get("TALKIE_GC_STRIDE", "1")))
        except ValueError:
            self.gc_stride = 1

        self._rope_cos: torch.Tensor | None = None
        self._rope_sin: torch.Tensor | None = None

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=None):
        self.gradient_checkpointing = enable

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, value):
        self.embed = value

    def _get_rope(
        self, seq_len: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target = max(seq_len, self.config.max_position_embeddings)
        if (
            self._rope_cos is None
            or self._rope_cos.shape[1] < target
            or self._rope_cos.device != device
        ):
            cos, sin = _precompute_rotary_embeddings(
                target,
                self.config.head_dim,
                self.config.rope_theta,
                device=device,
            )
            self._rope_cos = cos
            self._rope_sin = sin
        return self._rope_cos[:, :target], self._rope_sin[:, :target]

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        if inputs_embeds is None:
            assert input_ids is not None
            x = self.embed(input_ids)
            seq_len = input_ids.shape[1]
            device = input_ids.device
        else:
            x = inputs_embeds
            seq_len = inputs_embeds.shape[1]
            device = inputs_embeds.device

        cos_table, sin_table = self._get_rope(seq_len, device)
        if position_ids is not None:
            cos_sin = _gather_rope_per_position(cos_table, sin_table, position_ids)
        else:
            cos_sin = (cos_table[:, :seq_len], sin_table[:, :seq_len])

        # FlashAttention varlen path is for packed-sequence training only.
        # During inference (HF generate, vLLM, etc.) we go through
        # ALL_ATTENTION_FUNCTIONS instead.
        cu_seqlens, max_seqlen = (None, None)
        if self.training and position_ids is not None and _HAS_FA:
            cu_seqlens = _cu_seqlens_from_position_ids(position_ids)
            max_seqlen = seq_len

        x = F.rms_norm(x, (x.shape[-1],))
        e_x = x
        for i, block in enumerate(self.blocks):
            if (
                self.gradient_checkpointing
                and self.training
                and (i % self.gc_stride == 0)
            ):
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    e_x,
                    x,
                    cos_sin,
                    cu_seqlens,
                    max_seqlen,
                    use_reentrant=False,
                )
            else:
                x = block(e_x, x, cos_sin, cu_seqlens, max_seqlen, **kwargs)
        x = F.rms_norm(x, (x.shape[-1],))

        if return_dict is False:
            return (x,)
        return BaseModelOutputWithPast(last_hidden_state=x)


class TalkieForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = TalkieConfig
    _no_split_modules = ["TalkieDecoderLayer"]
    _supports_gradient_checkpointing = True
    supports_gradient_checkpointing = True
    _supports_attention_backend = True
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def __init__(self, config: TalkieConfig):
        super().__init__(config)
        self.model = TalkieModel(config)
        self.lm_head = nn.Parameter(
            torch.zeros(config.vocab_size, config.hidden_size)
        )
        self.lm_head_gain = TalkieWeightGain()

        self.post_init()

    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func=None):
        self.model.gradient_checkpointing = enable

    def _get_rope(self, seq_len: int, device: torch.device):
        # Backwards-compat shim for inference/fast_generate.py — RoPE tables
        # now live on the inner TalkieModel.
        return self.model._get_rope(seq_len, device)

    def get_input_embeddings(self):
        return self.model.embed

    def set_input_embeddings(self, value):
        self.model.embed = value

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[CausalLMOutputWithPast, Tuple]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            return_dict=False,
        )
        hidden_states = outputs[0]

        loss = None
        if labels is not None and _HAS_LIGER:
            shift_hidden = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            scaled_weight = self.lm_head_gain(self.lm_head)
            loss_fn = LigerFusedLinearCrossEntropyLoss(return_token_accuracy=True)
            res = loss_fn(
                scaled_weight,
                shift_hidden.view(-1, shift_hidden.size(-1)),
                shift_labels.view(-1),
            )
            return TalkieCausalLMOutput(
                loss=res.loss, logits=None, token_accuracy=res.token_accuracy,
            )

        logits = F.linear(hidden_states, self.lm_head_gain(self.lm_head))
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        else:
            logits = logits.float()

        return CausalLMOutputWithPast(loss=loss, logits=logits)
