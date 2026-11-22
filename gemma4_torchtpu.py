"""
Native PyTorch / TorchTPU implementation of Google Gemma 4 (E2B-IT) Causal LM.

Designed for 100% weight-key and mathematical parity with:
1. Hugging Face `google/gemma-4-E2B-it` (`model.safetensors` keys `model.language_model.*`)
2. Google Tunix `tunix.models.gemma4.model.Gemma4` (PLE table, KV-shared layers 15..34,
   proportional global RoPE + sliding local RoPE, Q/K/V RMSNorm, and logit softcapping)
3. Hugging Face `peft` (`LoraConfig`, `get_peft_model`, `merge_and_unload`)
4. Google Cloud `torch_tpu` (`torch.device("tpu")`, Strict Eager, DeferAndFuse, and `torch.compile(backend="tpu")`)
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from safetensors import safe_open
from safetensors.torch import save_file


def get_torchtpu_device(preferred: Optional[str] = None) -> torch.device:
    """Returns torch.device('tpu') when torch_tpu is available, else falls back cleanly."""
    if preferred == "cpu":
        return torch.device("cpu")
    try:
        import torch_tpu  # noqa: F401
        return torch.device("tpu")
    except ImportError:
        if preferred == "tpu":
            raise RuntimeError("Requested device='tpu' but torch_tpu is not installed.")
        return torch.device("cpu")


@dataclasses.dataclass
class Gemma4TextConfig:
    vocab_size: int = 262144
    vocab_size_per_layer_input: int = 262144
    hidden_size: int = 1536
    intermediate_size: int = 6144
    override_kv_shared_ffw_hidden: int = 12288
    num_hidden_layers: int = 35
    num_kv_shared_layers: int = 20
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    hidden_size_per_layer_input: int = 256
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    sliding_window: int = 512
    local_rope_theta: float = 10000.0
    global_rope_theta: float = 1000000.0
    global_partial_rotary_factor: float = 0.25
    pad_token_id: int = 0
    bos_token_id: int = 2
    eos_token_id: int = 1
    use_cache: bool = False
    tie_word_embeddings: bool = True
    model_type: str = "gemma4_text"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "Gemma4TextConfig":
        cfg_path = os.path.join(model_dir, "config.json")
        if not os.path.exists(cfg_path):
            return cls()
        with open(cfg_path, "r") as f:
            raw = json.load(f)
        tc = raw.get("text_config", raw)
        num_layers = tc.get("num_hidden_layers", 35)
        num_shared = tc.get("num_kv_shared_layers", 20)
        inter = tc.get("intermediate_size", 6144)
        double_wide = tc.get("use_double_wide_mlp", True)
        return cls(
            vocab_size=tc.get("vocab_size", 262144),
            vocab_size_per_layer_input=tc.get("vocab_size_per_layer_input", 262144),
            hidden_size=tc.get("hidden_size", 1536),
            intermediate_size=inter,
            override_kv_shared_ffw_hidden=inter * 2 if double_wide else inter,
            num_hidden_layers=num_layers,
            num_kv_shared_layers=num_shared,
            num_attention_heads=tc.get("num_attention_heads", 8),
            num_key_value_heads=tc.get("num_key_value_heads", 1),
            head_dim=tc.get("head_dim", 256),
            global_head_dim=tc.get("global_head_dim", 512),
            hidden_size_per_layer_input=tc.get("hidden_size_per_layer_input", 256),
            rms_norm_eps=tc.get("rms_norm_eps", 1e-6),
            final_logit_softcapping=tc.get("final_logit_softcapping", 30.0),
            sliding_window=tc.get("sliding_window", 512),
        )

    @classmethod
    def dry_run_tiny(cls) -> "Gemma4TextConfig":
        """Scaled-down 2-layer config for fast local CPU dry-run verification."""
        return cls(
            vocab_size=1024,
            vocab_size_per_layer_input=1024,
            hidden_size=64,
            intermediate_size=128,
            override_kv_shared_ffw_hidden=256,
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            global_head_dim=32,
            hidden_size_per_layer_input=16,
            sliding_window=64,
        )


class Gemma4RMSNorm(nn.Module):
    """RMSNorm matching Gemma 4 safetensors key `.weight` and Tunix float32 variance."""

    def __init__(self, dim: int, eps: float = 1e-6, with_scale: bool = True, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim, dtype=dtype))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_fp32 = x.to(torch.float32)
        var = x_fp32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_fp32 * torch.rsqrt(var + self.eps)
        if self.with_scale and self.weight is not None:
            normed = normed * self.weight.to(torch.float32)
        return normed.to(orig_dtype)


def apply_gemma4_rope(
    inputs: torch.Tensor,
    positions: torch.Tensor,
    base_frequency: float,
    rope_proportion: float = 1.0,
) -> torch.Tensor:
    """Applies Gemma 4 proportional RoPE to inputs of shape [B, L, N, H]."""
    head_dim = inputs.shape[-1]
    rope_angles = int(rope_proportion * head_dim // 2)
    nope_angles = head_dim // 2 - rope_angles

    device = inputs.device
    freq_exponents = (2.0 / head_dim) * torch.arange(0, rope_angles, dtype=torch.float32, device=device)
    timescale = torch.pow(float(base_frequency), freq_exponents)
    inv_timescale = 1.0 / timescale
    if nope_angles > 0:
        inv_timescale = F.pad(inv_timescale, (0, nope_angles), value=0.0)

    # positions: [B, L] -> sinusoid_inp: [B, L, 1, H//2]
    sinusoid_inp = positions.to(torch.float32).unsqueeze(-1) * inv_timescale.unsqueeze(0).unsqueeze(0)
    sinusoid_inp = sinusoid_inp.unsqueeze(2)

    sin = torch.sin(sinusoid_inp)
    cos = torch.cos(sinusoid_inp)

    half = head_dim // 2
    first_half = inputs[..., :half].to(torch.float32)
    second_half = inputs[..., half:].to(torch.float32)

    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin
    out = torch.cat([first_part, second_part], dim=-1)
    return out.to(inputs.dtype)


class Gemma4Attention(nn.Module):
    """Gemma 4 self-attention supporting both local sliding and global layers."""

    def __init__(self, config: Gemma4TextConfig, layer_idx: int, is_global: bool, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_global = is_global
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.global_head_dim if is_global else config.head_dim
        self.rope_base = config.global_rope_theta if is_global else config.local_rope_theta
        self.rope_proportion = config.global_partial_rotary_factor if is_global else 1.0

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False, dtype=dtype)

        self.q_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.k_norm = Gemma4RMSNorm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: torch.Tensor,
        shared_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
        q = self.q_norm(q)
        q = apply_gemma4_rope(q, positions, self.rope_base, self.rope_proportion)

        if shared_kv is not None:
            k, v = shared_kv
        else:
            k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
            v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
            # Value RMSNorm without learned scale (matches Tunix lines 909-910)
            v_fp32 = v.to(torch.float32)
            v_var = v_fp32.pow(2).mean(dim=-1, keepdim=True)
            v = (v_fp32 * torch.rsqrt(v_var + self.config.rms_norm_eps)).to(x.dtype)
            k = self.k_norm(k)
            k = apply_gemma4_rope(k, positions, self.rope_base, self.rope_proportion)

        # Transpose to [B, N, T, H] for contiguous TPU matmul
        q_t = q.transpose(1, 2).contiguous()  # [B, N, T, H]
        k_t = k.transpose(1, 2).contiguous()  # [B, K, T, H]
        v_t = v.transpose(1, 2).contiguous()  # [B, K, T, H]

        if self.num_kv_heads == 1 and self.num_heads > 1:
            k_t = k_t.expand(B, self.num_heads, T, self.head_dim)
            v_t = v_t.expand(B, self.num_heads, T, self.head_dim)
        elif self.num_kv_heads != self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k_t = k_t.repeat_interleave(repeats, dim=1)
            v_t = v_t.repeat_interleave(repeats, dim=1)

        # Unscaled dot product (Q and K are already RMSNormed)
        scores = torch.matmul(q_t, k_t.transpose(-2, -1))  # [B, N, T, T]
        scores = scores.masked_fill(~attn_mask, -2.3819763e38)
        probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_out = torch.matmul(probs, v_t)  # [B, N, T, H]
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.num_heads * self.head_dim)
        out = self.o_proj(attn_out)
        return out, (k, v)


class Gemma4MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Gemma4DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        is_global: bool,
        is_kv_shared: bool,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_global = is_global
        self.is_kv_shared = is_kv_shared

        self.input_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.self_attn = Gemma4Attention(config, layer_idx=layer_idx, is_global=is_global, dtype=dtype)
        self.post_attention_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

        self.pre_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        mlp_dim = config.override_kv_shared_ffw_hidden if is_kv_shared else config.intermediate_size
        self.mlp = Gemma4MLP(config.hidden_size, mlp_dim, dtype=dtype)
        self.post_feedforward_layernorm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

        if config.hidden_size_per_layer_input > 0:
            self.per_layer_input_gate = nn.Linear(
                config.hidden_size, config.hidden_size_per_layer_input, bias=False, dtype=dtype
            )
            self.per_layer_projection = nn.Linear(
                config.hidden_size_per_layer_input, config.hidden_size, bias=False, dtype=dtype
            )
            self.post_per_layer_input_norm = Gemma4RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
            )

        self.layer_scalar = nn.Parameter(torch.ones(1, dtype=dtype))

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: torch.Tensor,
        per_layer_input: Optional[torch.Tensor] = None,
        shared_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        norm_x = self.input_layernorm(x)
        attn_out, kv = self.self_attn(norm_x, positions, attn_mask, shared_kv=shared_kv)
        attn_out = self.post_attention_layernorm(attn_out) + x

        norm_ffw = self.pre_feedforward_layernorm(attn_out)
        ffw = self.post_feedforward_layernorm(self.mlp(norm_ffw)) + attn_out

        if self.config.hidden_size_per_layer_input > 0 and per_layer_input is not None:
            gated = F.gelu(self.per_layer_input_gate(ffw), approximate="tanh") * per_layer_input
            proj = self.post_per_layer_input_norm(self.per_layer_projection(gated))
            ffw = ffw + proj

        ffw = ffw * self.layer_scalar
        return ffw, kv


class Gemma4LanguageModel(nn.Module):
    """Inner `model.language_model` matching exact safetensors key hierarchy."""

    def __init__(self, config: Gemma4TextConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)

        if config.hidden_size_per_layer_input > 0:
            total_ple_dim = config.num_hidden_layers * config.hidden_size_per_layer_input
            self.embed_tokens_per_layer = nn.Embedding(
                config.vocab_size_per_layer_input, total_ple_dim, dtype=dtype
            )
            self.per_layer_model_projection = nn.Linear(
                config.hidden_size, total_ple_dim, bias=False, dtype=dtype
            )
            self.per_layer_projection_norm = Gemma4RMSNorm(
                config.hidden_size_per_layer_input, eps=config.rms_norm_eps, dtype=dtype
            )

        num_unshared = config.num_hidden_layers - config.num_kv_shared_layers
        self.kv_sharing_map: List[int] = []
        self.layers = nn.ModuleList()

        for i in range(config.num_hidden_layers):
            # 5-layer pattern: 4 sliding, 1 global (layers 4, 9, 14, 19, 24, 29, 34)
            is_global = (i % 5) == 4
            if i < num_unshared:
                shared_origin = i
            else:
                shared_origin = (num_unshared - 1) if is_global else (num_unshared - 2)
            self.kv_sharing_map.append(shared_origin)
            is_kv_shared = shared_origin != i
            self.layers.append(
                Gemma4DecoderLayer(
                    config=config,
                    layer_idx=i,
                    is_global=is_global,
                    is_kv_shared=is_kv_shared,
                    dtype=dtype,
                )
            )

        self.shared_origins = {origin for i, origin in enumerate(self.kv_sharing_map) if i != origin}
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def encode_per_layer_inputs(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        L = self.config.num_hidden_layers
        P = self.config.hidden_size_per_layer_input

        safe_ids = torch.where(
            (input_ids >= 0) & (input_ids < self.config.vocab_size_per_layer_input),
            input_ids,
            torch.zeros_like(input_ids),
        )
        # Tunix scales per_layer_model_projection by embed_dim ** -0.5
        proj_scale = float(self.config.hidden_size) ** -0.5
        x_proj = (self.per_layer_model_projection(x) * proj_scale).view(B, T, L, P)
        x_proj = self.per_layer_projection_norm(x_proj)

        y_ple = self.embed_tokens_per_layer(safe_ids).view(B, T, L, P)
        y_ple = y_ple * math.sqrt(P)
        return (x_proj + y_ple) * (2.0 ** -0.5)


class Gemma4CausalLMOutput:
    def __init__(self, loss: Optional[torch.Tensor], logits: torch.Tensor):
        self.loss = loss
        self.logits = logits


class Gemma4ModelWrapper(nn.Module):
    """Wraps `language_model` under `model.language_model.*` to match HF safetensors keys."""

    def __init__(self, config: Gemma4TextConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.language_model = Gemma4LanguageModel(config, dtype=dtype)


class Gemma4ForCausalLMTorchTPU(nn.Module):
    """
    Top-level Gemma 4 Causal LM compatible with Hugging Face `peft` (`get_peft_model`)
    and `torch_tpu` (`torch.device("tpu")` & `torch.compile(backend="tpu")`).
    """

    def __init__(self, config: Gemma4TextConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.config = config
        self.model = Gemma4ModelWrapper(config, dtype=dtype)
        self.gradient_checkpointing = True

    @property
    def language_model(self) -> Gemma4LanguageModel:
        return self.model.language_model

    def get_input_embeddings(self) -> nn.Module:
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value: nn.Module):
        self.model.language_model.embed_tokens = value

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    @staticmethod
    def build_masks_and_positions(
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        sliding_window: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Builds static-shape [B, 1, T, T] causal and sliding-window boolean masks + positions."""
        B, T = input_ids.shape
        device = input_ids.device
        if attention_mask is None:
            valid_mask = input_ids != 0
        else:
            valid_mask = attention_mask.to(torch.bool)

        # Positions matching Tunix `build_positions_from_mask`: cumsum(valid) - 1, clamped >= 0
        positions = (torch.cumsum(valid_mask.to(torch.long), dim=-1) - 1).clamp(min=0)

        idx = torch.arange(T, device=device)
        causal_2d = idx.unsqueeze(1) >= idx.unsqueeze(0)  # [T, T]
        # Both query and key positions must be valid tokens (or self-attend on diagonal to avoid NaN softmax)
        valid_kv = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)  # [B, T, T]
        diag = torch.eye(T, dtype=torch.bool, device=device).unsqueeze(0)  # [1, T, T]
        global_mask = ((causal_2d.unsqueeze(0) & valid_kv) | diag).unsqueeze(1).contiguous()  # [B, 1, T, T]

        dist = idx.unsqueeze(1) - idx.unsqueeze(0)
        window_2d = (dist >= 0) & (dist < sliding_window)
        sliding_mask = ((window_2d.unsqueeze(0) & valid_kv) | diag).unsqueeze(1).contiguous()  # [B, 1, T, T]
        return positions.contiguous(), global_mask, sliding_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        return_logits: bool = True,
        last_token_idx: Optional[int] = None,
        **kwargs,
    ) -> Gemma4CausalLMOutput:
        lm = self.model.language_model
        cfg = self.config

        pos, global_mask, sliding_mask = self.build_masks_and_positions(
            input_ids, attention_mask, cfg.sliding_window
        )
        if positions is not None:
            pos = positions

        x = lm.embed_tokens(input_ids) * math.sqrt(cfg.hidden_size)
        per_layer_inputs = None
        if cfg.hidden_size_per_layer_input > 0:
            per_layer_inputs = lm.encode_per_layer_inputs(x, input_ids)

        transient_kvs: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for i, layer in enumerate(lm.layers):
            origin = lm.kv_sharing_map[i]
            shared_kv = transient_kvs.get(origin) if origin != i else None
            layer_mask = global_mask if layer.is_global else sliding_mask
            pli = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None

            if self.training and self.gradient_checkpointing and i not in lm.shared_origins:
                def run_layer(module, h, p, m, pl, skv):
                    out_h, _ = module(h, p, m, per_layer_input=pl, shared_kv=skv)
                    return out_h
                x = checkpoint(run_layer, layer, x, pos, layer_mask, pli, shared_kv, use_reentrant=False)
                kv = None
            else:
                x, kv = layer(x, pos, layer_mask, per_layer_input=pli, shared_kv=shared_kv)
            if i in lm.shared_origins and kv is not None:
                transient_kvs[i] = kv

        x = lm.norm(x)
        cap = float(cfg.final_logit_softcapping) if (cfg.final_logit_softcapping and cfg.final_logit_softcapping > 0) else None

        if last_token_idx is not None:
            x_last = x[:, last_token_idx : last_token_idx + 1, :]
            logits_last = F.linear(x_last, lm.embed_tokens.weight).to(torch.float32)
            if cap is not None:
                logits_last = torch.tanh(logits_last / cap) * cap
            return Gemma4CausalLMOutput(loss=None, logits=logits_last)

        # Chunked Vocabulary Cross-Entropy to avoid materializing 4.3GB x 3 float32 [B, T, 262144] logits in HBM
        if loss_mask is not None:
            B = x.shape[0]
            total_masked_loss = torch.zeros((), dtype=torch.float32, device=x.device)
            total_tokens = loss_mask[:, 1:].sum().clamp(min=1.0)
            for b in range(B):
                logits_b = F.linear(x[b, :-1, :], lm.embed_tokens.weight).to(torch.float32)
                if cap is not None:
                    logits_b = torch.tanh(logits_b / cap) * cap
                targets_b = input_ids[b, 1:].contiguous()
                mask_b = loss_mask[b, 1:].contiguous()
                ce_b = F.cross_entropy(logits_b, targets_b, reduction="none")
                total_masked_loss = total_masked_loss + (ce_b * mask_b).sum()
            loss = total_masked_loss / total_tokens
            if not return_logits:
                return Gemma4CausalLMOutput(loss=loss, logits=x)

        logits = F.linear(x, lm.embed_tokens.weight).to(torch.float32)
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, cfg.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return Gemma4CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate_response(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 128,
        eos_token_ids: Tuple[int, ...] = (1, 106),
        temperature: float = 0.0,
        device: Optional[torch.device] = None,
        static_pad_len: int = 256,
    ) -> List[int]:
        """
        Static-shape autoregressive generation on TPU or CPU.
        Pads `[1, cur_len]` to a fixed `[1, static_pad_len]` bucket so XLA/TorchTPU
        compiles once and executes every decode step with zero dynamic-shape recompilations.
        """
        if device is None:
            device = next(self.parameters()).device
        generated: List[int] = list(prompt_ids)
        bucket_len = max(static_pad_len, len(prompt_ids) + max_new_tokens)

        # Pre-allocate static buffers on device
        cur_ids = torch.zeros((1, bucket_len), dtype=torch.long, device=device)
        cur_mask = torch.zeros((1, bucket_len), dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            cur_len = len(generated)
            cur_ids.zero_()
            cur_mask.zero_()
            cur_ids[0, :cur_len] = torch.tensor(generated, dtype=torch.long, device=device)
            cur_mask[0, :cur_len] = 1

            out = self.forward(
                input_ids=cur_ids,
                attention_mask=cur_mask,
                last_token_idx=cur_len - 1,
            )
            next_logits = out.logits[0, 0, :]
            if temperature <= 1e-5:
                next_id = int(torch.argmax(next_logits).item())
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())
            if next_id in eos_token_ids:
                break
            generated.append(next_id)
        return generated[len(prompt_ids) :]

    @classmethod
    def from_safetensors(
        cls,
        model_dir: str,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "Gemma4ForCausalLMTorchTPU":
        """Loads `model.language_model.*` weights directly from `model.safetensors`."""
        config = Gemma4TextConfig.from_pretrained(model_dir)
        # Initialize on meta device first to avoid allocating 9.5GB twice in RAM
        with torch.device("meta"):
            model = cls(config, dtype=dtype)

        st_path = os.path.join(model_dir, "model.safetensors")
        state_dict: Dict[str, torch.Tensor] = {}
        skipped_mm = 0
        with safe_open(st_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.startswith("model.language_model."):
                    state_dict[k] = f.get_tensor(k).to(dtype=dtype)
                else:
                    skipped_mm += 1

        missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
        if missing:
            raise RuntimeError(f"Missing language model keys when loading {st_path}: {missing[:10]}")
        print(
            f"[Gemma4-TorchTPU] Loaded {len(state_dict)} language_model tensors "
            f"(skipped {skipped_mm} vision/audio tensors) -> moving to {device}..."
        )
        return model.to(device)

    def save_merged_safetensors(self, base_model_dir: str, output_dir: str):
        """Saves merged language_model weights + tokenizer/config files to `output_dir`."""
        os.makedirs(output_dir, exist_ok=True)
        state_dict = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        out_st = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, out_st)
        for fname in [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "generation_config.json",
            "chat_template.jinja",
        ]:
            src = os.path.join(base_model_dir, fname)
            if os.path.exists(src):
                with open(src, "rb") as fsrc, open(os.path.join(output_dir, fname), "wb") as fdst:
                    fdst.write(fsrc.read())
        print(f"[Gemma4-TorchTPU] Saved {len(state_dict)} tensors to {out_st}")
