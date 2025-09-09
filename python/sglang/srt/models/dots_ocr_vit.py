from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from sglang.srt.configs.dots_ocr import DotsOCRVisionConfig
from sglang.srt.layers.activation import QuickGELU, SiluAndMul
from sglang.srt.layers.attention.vision import VisionAttention, apply_rotary_pos_emb
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
)
from torch.nn import LayerNorm
from transformers.modeling_utils import PreTrainedModel

try:
    from flash_attn import flash_attn_varlen_func  # pyright: ignore[reportMissingImports]

    flash_attn_available = True
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_available = False


class VisionRotaryEmbedding(nn.Module):
    freqs_cache: torch.Tensor
    inv_freq: torch.Tensor

    def __init__(self, dim: int, theta: float = 10000.0, max_seqlen: int = 8192) -> None:
        super().__init__()
        self.dim = dim
        self.max_seqlen = max_seqlen
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        freqs_cache = self._compute_freqs_cache()
        self.register_buffer("freqs_cache", freqs_cache, persistent=False)

    def _compute_freqs_cache(self) -> torch.Tensor:
        seq = torch.arange(self.max_seqlen, dtype=torch.float)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs

    def forward(self, seqlen: int) -> torch.Tensor:
        if seqlen <= self.max_seqlen:
            return self.freqs_cache[:seqlen]

        # Fallback to dynamic computation for sequences longer than cache
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchMerger(nn.Module):
    def __init__(
        self, dim: int, context_dim: int, spatial_merge_size: int = 2, pre_norm: str = "layernorm", *args, **kwargs
    ) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.pre_norm = pre_norm
        if self.pre_norm == "layernorm":
            self.ln_q = LayerNorm(context_dim, eps=1e-6)
        elif self.pre_norm == "rmsnorm":
            self.ln_q = RMSNorm(context_dim, eps=1e-6)
        else:
            print("no norm in patch merger")

        self.mlp = nn.ModuleList(
            [
                ColumnParallelLinear(
                    self.hidden_size,
                    self.hidden_size,
                    bias=True,
                    quant_config=getattr(kwargs.get("config"), "quant_config", None),
                ),
                QuickGELU(),
                RowParallelLinear(
                    self.hidden_size,
                    dim,
                    bias=True,
                    quant_config=getattr(kwargs.get("config"), "quant_config", None),
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = self.ln_q(x).view(-1, self.hidden_size)
        else:
            x = x.view(-1, self.hidden_size)

        mlp_fc1, mlp_act, mlp_fc2 = self.mlp
        x_parallel, _ = mlp_fc1(x)
        x_parallel = mlp_act(x_parallel)
        out, _ = mlp_fc2(x_parallel)
        return out


class DotsSwiGLUFFN(nn.Module):
    def __init__(self, config: DotsOCRVisionConfig) -> None:
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.embed_dim
        bias = config.use_bias

        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=in_features,
            output_sizes=[hidden_features] * 2,
            bias=bias,
            quant_config=getattr(config, "quant_config", None),
        )
        self.down_proj = RowParallelLinear(
            hidden_features,
            in_features,
            bias=bias,
            quant_config=getattr(config, "quant_config", None),
        )

        self.act = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act(gate_up)
        x, _ = self.down_proj(x)
        return x


class DotsPatchEmbed(nn.Module):
    def __init__(self, config: DotsOCRVisionConfig) -> None:
        super().__init__()
        self.num_channels = config.num_channels
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.embed_dim = config.embed_dim
        self.config = config
        self.proj = nn.Conv2d(
            config.num_channels,
            config.embed_dim,
            kernel_size=(config.patch_size, config.patch_size),
            stride=(config.patch_size, config.patch_size),
        )
        self.norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(
            -1,
            self.num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )[:, :, 0]
        x = self.proj(x).view(-1, self.embed_dim)
        x = self.norm(x)
        return x


class DotsViTPreprocessor(nn.Module):
    def __init__(self, config: DotsOCRVisionConfig) -> None:
        super().__init__()
        self.patch_h = config.patch_size
        self.patch_w = config.patch_size
        self.embed_dim = config.embed_dim
        self.config = config
        self.patchifier = DotsPatchEmbed(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patchifier(x)
        return tokens


class DotsVisionBlock(nn.Module):
    def __init__(self, config: DotsOCRVisionConfig) -> None:
        super().__init__()
        self.attn = VisionAttention(
            embed_dim=config.embed_dim,
            num_heads=config.num_attention_heads,
            projection_size=config.embed_dim,
            use_qkv_parallel=True,
            proj_bias=config.use_bias,
            qkv_bias=config.use_bias,
        )
        self.norm1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.mlp = DotsSwiGLUFFN(config)
        self.norm2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        ).squeeze(0)

        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class DotsVisionBlock2(nn.Module):
    def __init__(self, config: DotsOCRVisionConfig) -> None:
        super().__init__()
        self.attn = FlashAttn(config, config.embed_dim, config.num_attention_heads, bias=config.use_bias)
        self.norm1 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)
        self.mlp = DotsSwiGLUFFN(config)
        self.norm2 = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            max_seqlen=max_seqlen,
        ).squeeze(0)

        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class FlashAttn(nn.Module):
    def __init__(self, config: DotsOCRVisionConfig, dim: int, num_heads: int = 16, bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv_proj = ColumnParallelLinear(dim, dim * 3, bias=bias)
        self.proj = RowParallelLinear(dim, dim, bias=bias)
        self.config = config
        self.is_causal = config.is_causal

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int,
    ) -> torch.Tensor:
        assert flash_attn_available, "flash attention is not available"

        seq_length = hidden_states.shape[0]
        q, k, v = (
            self.qkv_proj(hidden_states)[0].reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        )  # 'shd'

        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            causal=self.is_causal,
        ).reshape(seq_length, -1)  # type: ignore
        attn_output = self.proj(attn_output)[0]

        return attn_output


class DotsVisionTransformer(PreTrainedModel):
    def __init__(self, config: DotsOCRVisionConfig) -> None:
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = DotsViTPreprocessor(config)

        head_dim = config.embed_dim // config.num_attention_heads

        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        _num_hidden_layers = config.num_hidden_layers
        self.blocks = nn.ModuleList([DotsVisionBlock2(config) for _ in range(_num_hidden_layers)])

        if self.config.post_norm:
            self.post_trunk_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

        self.merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
            init_merger_std=self.config.init_merger_std,
            config=config,
        )

    def load_state_dict(
        self,
        state_dict: torch.utils.checkpoint.Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        mapped_state_dict = self._map_dots_to_sglang_keys(state_dict)
        return super().load_state_dict(mapped_state_dict, strict=strict)

    @staticmethod
    def _map_dots_to_sglang_keys(state_dict):
        """
        Maps Dots keys to SGLang keys.
        - Converts 'blocks.{i}.attn.qkv.weight' to 'blocks.{i}.attn.qkv_proj.weight'
        - Fuses fc1 (gate) and fc3 (up) into gate_up_proj, maps fc2 to down_proj
        """
        import torch

        mapped_dict = {}
        gate_weights = {}
        up_weights = {}

        for key, value in state_dict.items():
            if ".attn.qkv." in key:
                new_key = key.replace(".attn.qkv.", ".attn.qkv_proj.")
                mapped_dict[new_key] = value
            elif ".mlp.fc1." in key:
                # Store gate projection weights
                gate_key = key.replace(".mlp.fc1.", ".mlp.gate_up_proj.")
                gate_weights[gate_key] = value
            elif ".mlp.fc3." in key:
                # Store up projection weights
                up_key = key.replace(".mlp.fc3.", ".mlp.gate_up_proj.")
                up_weights[up_key] = value
            elif ".mlp.fc2." in key:
                # Map down projection directly
                new_key = key.replace(".mlp.fc2.", ".mlp.down_proj.")
                mapped_dict[new_key] = value
            else:
                mapped_dict[key] = value

        # Fuse gate and up projections
        for gate_key, gate_value in gate_weights.items():
            up_key = gate_key
            if up_key in up_weights:
                up_value = up_weights[up_key]
                fused_weight = torch.cat([gate_value, up_value], dim=0)
                mapped_dict[gate_key] = fused_weight

        return mapped_dict

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)  # type: ignore
            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)  # type: ignore
            hpos_ids = (
                hpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            wpos_ids = (
                wpos_ids.reshape(
                    h // self.spatial_merge_size,
                    self.spatial_merge_size,
                    w // self.spatial_merge_size,
                    self.spatial_merge_size,
                )
                .permute(0, 2, 1, 3)
                .flatten()
            )
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))  # type: ignore
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        cos = rotary_pos_emb.cos().repeat(1, 2)
        sin = rotary_pos_emb.sin().repeat(1, 2)
        return cos, sin

    def _blk_forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        max_seqlen: int,
    ) -> torch.Tensor:
        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                max_seqlen=max_seqlen,
            )

        if self.config.post_norm:
            hidden_states = self.post_trunk_norm(hidden_states)

        hidden_states = self.merger(hidden_states)
        return hidden_states

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(hidden_states.device)
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        position_embeddings = self.rot_pos_emb(grid_thw)
        return self._blk_forward(hidden_states, cu_seqlens, position_embeddings, max_seqlen)  # type: ignore
