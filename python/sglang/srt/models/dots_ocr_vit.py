import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from sglang.srt.configs.dots_ocr import DotsOCRVisionConfig
from sglang.srt.layers.attention.vision import VisionAttention
from sglang.srt.layers.layernorm import RMSNorm
from torch.nn import LayerNorm
from transformers.modeling_utils import PreTrainedModel


class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchMerger(nn.Module):
    def __init__(
        self,
        dim: int,
        context_dim: int,
        spatial_merge_size: int = 2,
        pre_norm="layernorm",
        init_merger_std=None,
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

        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm:
            x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        else:
            x = self.mlp(x.view(-1, self.hidden_size))
        return x


class DotsSwiGLUFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_features = config.intermediate_size
        in_features = config.embed_dim
        bias = config.use_bias

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(x)) * self.fc3(x)
        x = self.fc2(x)
        return x


class DotsPatchEmbed(nn.Module):
    def __init__(self, config):
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

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:
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
    def __init__(self, config):
        super().__init__()
        self.patch_h = config.patch_size
        self.patch_w = config.patch_size
        self.embed_dim = config.embed_dim
        self.config = config
        self.patchifier = DotsPatchEmbed(config)

    def forward(self, x: torch.Tensor, grid_thw=None) -> torch.Tensor:
        tokens = self.patchifier(x, grid_thw)
        return tokens


class DotsVisionBlock(nn.Module):
    def __init__(self, config, attn_implementation: str = "flash_attention_2"):
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
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        ).squeeze(0)

        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class DotsVisionTransformer(PreTrainedModel):
    def __init__(self, config: DotsOCRVisionConfig) -> None:
        super().__init__(config)
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = DotsViTPreprocessor(config)
        self._init_weights(self.patch_embed.patchifier.proj)

        head_dim = config.embed_dim // config.num_attention_heads

        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        _num_hidden_layers = config.num_hidden_layers
        self.blocks = nn.ModuleList(
            [
                DotsVisionBlock(config, config.attn_implementation)
                for _ in range(_num_hidden_layers)
            ]
        )

        if self.config.post_norm:
            self.post_trunk_norm = RMSNorm(config.embed_dim, eps=config.rms_norm_eps)

        self.merger = PatchMerger(
            dim=config.hidden_size,
            context_dim=config.embed_dim,
            spatial_merge_size=config.spatial_merge_size,
            init_merger_std=self.config.init_merger_std,
        )

    @staticmethod
    def _map_dots_to_sglang_keys(state_dict):
        """
        Maps Dots VisionAttention keys to SGLang VisionAttention keys.
        Converts 'blocks.{i}.attn.qkv.weight' to 'blocks.{i}.attn.qkv_proj.weight'
        """
        mapped_dict = {}
        for key, value in state_dict.items():
            if ".attn.qkv." in key:
                new_key = key.replace(".attn.qkv.", ".attn.qkv_proj.")
                mapped_dict[new_key] = value
            else:
                mapped_dict[key] = value
        return mapped_dict

    def load_state_dict(self, state_dict, strict=True):
        """Override to apply key mapping before loading state dict."""
        mapped_state_dict = self._map_dots_to_sglang_keys(state_dict)
        return super().load_state_dict(mapped_state_dict, strict=strict)

    def get_pos_ids_by_grid(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1))

        return pos_ids

    def rot_pos_emb(self, grid_thw):
        pos_ids = self.get_pos_ids_by_grid(grid_thw)
        pos_ids = torch.cat(pos_ids, dim=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states, grid_thw)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        # DO NOT DO THIS! FIGURE OUT WHY GRID_THW IS ON CPU
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0).to(hidden_states.device)

        cos = rotary_pos_emb.cos().repeat(1, 2)
        sin = rotary_pos_emb.sin().repeat(1, 2)
        position_embeddings = (cos, sin)

        for blk in self.blocks:
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )

        if self.config.post_norm:
            hidden_states = self.post_trunk_norm(hidden_states)

        hidden_states = self.merger(hidden_states)
        return hidden_states
