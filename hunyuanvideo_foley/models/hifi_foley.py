from typing import List, Tuple, Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .nn.activation_layers import SwiGLU, get_activation_layer
from .nn.attn_layers import apply_rotary_emb, attention
from .nn.embed_layers import TimestepEmbedder, ConditionProjection, PatchEmbed1D
from .nn.mlp_layers import MLP, ConvMLP, FinalLayer1D, ChannelLastConv1d
from .nn.modulate_layers import ModulateDiT, ckpt_wrapper, apply_gate, modulate
from .nn.norm_layers import get_norm_layer
from .nn.posemb_layers import get_nd_rotary_pos_embed

def interleave_two_sequences(x1: torch.Tensor, x2: torch.Tensor):
    # [B, N1, H, C] & [B, N2, H, C]
    B, N1, H, C = x1.shape
    B, N2, H, C = x2.shape
    assert x1.ndim == x2.ndim == 4

    if N1 != N2:
        x2 = x2.view(B, N2, -1).transpose(1, 2)
        x2 = F.interpolate(x2, size=(N1), mode="nearest-exact")
        x2 = x2.transpose(1, 2).view(B, N1, H, C)
    x = torch.stack((x1, x2), dim=2)
    x = x.reshape(B, N1 * 2, H, C)
    return x

def decouple_interleaved_two_sequences(x: torch.Tensor, len1: int, len2: int):
    B, N, H, C = x.shape
    assert N % 2 == 0 and N // 2 == len1

    x = x.reshape(B, -1, 2, H, C)
    x1 = x[:, :, 0]
    x2 = x[:, :, 1]
    if x2.shape[1] != len2:
        x2 = x2.view(B, len1, H * C).transpose(1, 2)
        x2 = F.interpolate(x2, size=(len2), mode="nearest-exact")
        x2 = x2.transpose(1, 2).view(B, len2, H, C)
    return x1, x2

class TwoStreamCABlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        attn_mode: str = "torch",
        reverse: bool = False,
        interleaved_audio_visual_rope: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.reverse = reverse
        self.attn_mode = attn_mode
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        head_dim = hidden_size // num_heads
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        self.interleaved_audio_visual_rope = interleaved_audio_visual_rope

        # Self attention for audio + visual
        self.audio_mod = ModulateDiT(hidden_size, factor=9, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.audio_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.audio_self_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.audio_self_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.audio_self_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.audio_self_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        # visual cond
        self.v_cond_mod = ModulateDiT(hidden_size, factor=9, act_layer=get_activation_layer("silu"), **factory_kwargs)
        self.v_cond_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.v_cond_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        self.v_cond_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.v_cond_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.v_cond_self_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        self.max_text_len = 100
        self.rope_dim_list = None
        
        # audio and video norm for cross attention with text
        self.audio_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.v_cond_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)

        # Cross attention: (video_audio) as query, text as key/value
        self.audio_cross_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.v_cond_cross_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.text_cross_kv = nn.Linear(hidden_size, hidden_size * 2, bias=qkv_bias, **factory_kwargs)
        
        self.audio_cross_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.v_cond_cross_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.text_cross_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.audio_cross_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)
        self.v_cond_cross_proj = nn.Linear(hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs)

        # MLPs
        self.audio_norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.audio_mlp = MLP(
            hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs
        )

        self.v_cond_norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.v_cond_mlp = MLP(
            hidden_size, mlp_hidden_dim, act_layer=get_activation_layer(mlp_act_type), bias=True, **factory_kwargs
        )

    def build_rope_for_text(self, text_len, head_dim, rope_dim_list=None):
        target_ndim = 1  # n-d RoPE
        rope_sizes = [text_len]
        
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        
        text_freqs_cos, text_freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list=rope_dim_list,
            start=rope_sizes,
            theta=10000,
            use_real=True,
            theta_rescale_factor=1.0,
        )
        return text_freqs_cos, text_freqs_sin

    def set_attn_mode(self, new_mode):
        if new_mode != "torch":
            raise NotImplementedError(f"Only support 'torch' mode, got {new_mode}.")
        self.attn_mode = new_mode

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        audio: torch.Tensor,
        cond: torch.Tensor,
        v_cond: torch.Tensor,
        attn_mask: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        v_freqs_cis: tuple = None,
        sync_vec: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get modulation parameters
        if sync_vec is not None:
            assert sync_vec.ndim == 3
            (audio_mod1_shift, audio_mod1_scale, audio_mod1_gate, 
             audio_mod2_shift, audio_mod2_scale, audio_mod2_gate, 
             audio_mod3_shift, audio_mod3_scale, audio_mod3_gate,
            ) = self.audio_mod(sync_vec).chunk(9, dim=-1)
        else:
            (audio_mod1_shift, audio_mod1_scale, audio_mod1_gate, 
             audio_mod2_shift, audio_mod2_scale, audio_mod2_gate,
             audio_mod3_shift, audio_mod3_scale, audio_mod3_gate,
            ) = self.audio_mod(vec).chunk(9, dim=-1)

        (
            v_cond_mod1_shift,
            v_cond_mod1_scale,
            v_cond_mod1_gate,
            v_cond_mod2_shift,
            v_cond_mod2_scale,
            v_cond_mod2_gate,
            v_cond_mod3_shift,
            v_cond_mod3_scale,
            v_cond_mod3_gate,
        ) = self.v_cond_mod(vec).chunk(9, dim=-1)
        
        # 1. Self Attention for audio + visual
        audio_modulated = self.audio_norm1(audio)
        audio_modulated = modulate(audio_modulated, shift=audio_mod1_shift, scale=audio_mod1_scale)
        audio_qkv = self.audio_self_attn_qkv(audio_modulated)
        audio_q, audio_k, audio_v = rearrange(audio_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        audio_q = self.audio_self_q_norm(audio_q).to(audio_v)
        audio_k = self.audio_self_k_norm(audio_k).to(audio_v)
        
        # Prepare visual cond for attention
        v_cond_modulated = self.v_cond_norm1(v_cond)
        v_cond_modulated = modulate(v_cond_modulated, shift=v_cond_mod1_shift, scale=v_cond_mod1_scale)
        v_cond_qkv = self.v_cond_attn_qkv(v_cond_modulated)
        v_cond_q, v_cond_k, v_cond_v = rearrange(v_cond_qkv, "B L (K H D) -> K B L H D", K=3, H=self.num_heads)
        v_cond_q = self.v_cond_attn_q_norm(v_cond_q).to(v_cond_v)
        v_cond_k = self.v_cond_attn_k_norm(v_cond_k).to(v_cond_v)
        
        # Apply RoPE if needed for audio and visual
        if freqs_cis is not None:
            if not self.interleaved_audio_visual_rope:
                audio_qq, audio_kk = apply_rotary_emb(audio_q, audio_k, freqs_cis, head_first=False)
                audio_q, audio_k = audio_qq, audio_kk
            else:
                ori_audio_len = audio_q.shape[1]
                ori_v_con_len = v_cond_q.shape[1]
                interleaved_audio_visual_q = interleave_two_sequences(audio_q, v_cond_q)
                interleaved_audio_visual_k = interleave_two_sequences(audio_k, v_cond_k)
                interleaved_audio_visual_qq, interleaved_audio_visual_kk = apply_rotary_emb(
                    interleaved_audio_visual_q, interleaved_audio_visual_k, freqs_cis, head_first=False
                )
                audio_qq, v_cond_qq = decouple_interleaved_two_sequences(
                    interleaved_audio_visual_qq, ori_audio_len, ori_v_con_len
                )
                audio_kk, v_cond_kk = decouple_interleaved_two_sequences(
                    interleaved_audio_visual_kk, ori_audio_len, ori_v_con_len
                )
                audio_q, audio_k = audio_qq, audio_kk
                v_cond_q, v_cond_k = v_cond_qq, v_cond_kk

        # Apply RoPE to visual if needed and not interleaved
        if v_freqs_cis is not None and not self.interleaved_audio_visual_rope:
            v_cond_qq, v_cond_kk = apply_rotary_emb(v_cond_q, v_cond_k, v_freqs_cis, head_first=False)
            v_cond_q, v_cond_k = v_cond_qq, v_cond_kk
        
        # Concatenate for self-attention
        q = torch.cat((v_cond_q, audio_q), dim=1)
        k = torch.cat((v_cond_k, audio_k), dim=1)
        v = torch.cat((v_cond_v, audio_v), dim=1)
        
        # Run self-attention
        attn = attention(q, k, v, mode=self.attn_mode, attn_mask=attn_mask, deterministic=self.deterministic)
        v_cond_attn, audio_attn = torch.split(attn, [v_cond.shape[1], audio.shape[1]], dim=1)
        
        # Apply self-attention output to audio and v_cond
        audio = audio + apply_gate(self.audio_self_proj(audio_attn), gate=audio_mod1_gate)
        v_cond = v_cond + apply_gate(self.v_cond_self_proj(v_cond_attn), gate=v_cond_mod1_gate)

        # 2. Cross Attention: (v_cond, audio) as query, text as key/value
        # audio, v_cond modulation
        audio_modulated = self.audio_norm2(audio)
        audio_modulated = modulate(audio_modulated, shift=audio_mod2_shift, scale=audio_mod2_scale)
        v_cond_modulated = self.v_cond_norm2(v_cond)
        v_cond_modulated = modulate(v_cond_modulated, shift=v_cond_mod2_shift, scale=v_cond_mod2_scale)

        # Prepare audio query
        audio_q = self.audio_cross_q(audio_modulated)
        audio_q = rearrange(audio_q, "B L (H D) -> B L H D", H=self.num_heads)
        audio_q = self.audio_cross_q_norm(audio_q)
        
        # Prepare v_cond query
        v_cond_q = self.v_cond_cross_q(v_cond_modulated)
        v_cond_q = rearrange(v_cond_q, "B L (H D) -> B L H D", H=self.num_heads)
        v_cond_q = self.v_cond_cross_q_norm(v_cond_q)

        # Prepare text key/value
        text_kv = self.text_cross_kv(cond)
        text_k, text_v = rearrange(text_kv, "B L (K H D) -> K B L H D", K=2, H=self.num_heads)
        text_k = self.text_cross_k_norm(text_k).to(text_v)
        
        # Apply RoPE to (v_cond, audio) query and text key if needed
        head_dim = self.hidden_size // self.num_heads
        audio_cross_freqs_cos, audio_cross_freqs_sin = self.build_rope_for_text(audio_q.shape[1], head_dim, rope_dim_list=self.rope_dim_list)
        audio_cross_freqs_cis = (audio_cross_freqs_cos.to(audio_q.device), audio_cross_freqs_sin.to(audio_q.device))
        audio_q = apply_rotary_emb(audio_q, audio_q, audio_cross_freqs_cis, head_first=False)[0]
        
        v_cond_cross_freqs_cos, v_cond_cross_freqs_sin = self.build_rope_for_text(v_cond_q.shape[1], head_dim, rope_dim_list=self.rope_dim_list)
        v_cond_cross_freqs_cis = (v_cond_cross_freqs_cos.to(v_cond_q.device), v_cond_cross_freqs_sin.to(v_cond_q.device))
        v_cond_q = apply_rotary_emb(v_cond_q, v_cond_q, v_cond_cross_freqs_cis, head_first=False)[0]

        text_len = text_k.shape[1]
        
        text_freqs_cos, text_freqs_sin = self.build_rope_for_text(text_len, head_dim, 
                                                                 rope_dim_list=self.rope_dim_list)
        text_freqs_cis = (text_freqs_cos.to(text_k.device), text_freqs_sin.to(text_k.device))
        text_k = apply_rotary_emb(text_k, text_k, text_freqs_cis, head_first=False)[1]
        
        # Concat v_cond and audio for cross-attention  
        v_cond_audio_q = torch.cat([v_cond_q, audio_q], dim=1)

        # Run cross-attention
        cross_attn = attention(v_cond_audio_q, text_k, text_v, mode=self.attn_mode, deterministic=self.deterministic)
        v_cond_cross_attn, audio_cross_attn = torch.split(cross_attn, [v_cond.shape[1], audio.shape[1]], dim=1)
        
        # Apply cross-attention output
        audio = audio + apply_gate(self.audio_cross_proj(audio_cross_attn), gate=audio_mod2_gate)
        v_cond = v_cond + apply_gate(self.v_cond_cross_proj(v_cond_cross_attn), gate=v_cond_mod2_gate)

        # 3. Apply MLPs
        audio = audio + apply_gate(
            self.audio_mlp(modulate(self.audio_norm3(audio), shift=audio_mod3_shift, scale=audio_mod3_scale)),
            gate=audio_mod3_gate,
        )
        
        # Apply visual MLP
        v_cond = v_cond + apply_gate(
            self.v_cond_mlp(modulate(self.v_cond_norm3(v_cond), shift=v_cond_mod3_shift, scale=v_cond_mod3_scale)),
            gate=v_cond_mod3_gate,
        )

        return audio, cond, v_cond

class SingleStreamBlock(nn.Module):

    def __init__(self, hidden_size: int,
                    num_heads: int,
                    mlp_ratio: float,
                    qk_norm_type: str = "rms",
                    dtype: Optional[torch.dtype] = None,
                    device: Optional[torch.device] = None,):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.modulation = ModulateDiT(
            hidden_size=hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.linear_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.linear1 = ChannelLastConv1d(hidden_size, hidden_size, kernel_size=3, padding=1, **factory_kwargs)
        self.linear2 = ConvMLP(hidden_size, hidden_size * mlp_ratio, kernel_size=3, padding=1, **factory_kwargs)
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.q_norm = nn.RMSNorm(hidden_size // num_heads)
        self.k_norm = nn.RMSNorm(hidden_size // num_heads)
        self.rearrange = Rearrange("B L (H D K) -> B H L D K", K=3, H=num_heads)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None):
        assert cond.ndim == 3, "Condition should be in shape of [B, T, D]"
        modulation = self.modulation(cond)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        x_norm1 = self.norm1(x) * (1 + scale_msa) + shift_msa

        qkv = self.linear_qkv(x_norm1)
        q, k, v = self.rearrange(qkv).chunk(3, dim=-1)
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)

        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = apply_rotary_emb(q, k, freqs_cis, head_first=True)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, 'b h n d -> b n (h d)').contiguous()

        x = x + apply_gate(self.linear1(out),gate=gate_msa)
        x_norm = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + apply_gate(self.linear2(x_norm), gate=gate_mlp)

        return x

class HunyuanVideoFoley(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        model_config,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        model_args = model_config.model_config.model_kwargs
        self.depth_triple_blocks = model_args.get("depth_triple_blocks", 19)
        self.depth_single_blocks = model_args.get("depth_single_blocks", 38)
        # Gradient checkpoint.
        self.gradient_checkpoint = False
        self.gradient_checkpoint_layers = None
        if self.gradient_checkpoint:
            assert self.gradient_checkpoint_layers <= self.depth_triple_blocks + self.depth_single_blocks, (
                f"Gradient checkpoint layers must be less or equal than the depth of the model. "
                f"Got gradient_checkpoint_layers={self.gradient_checkpoint_layers} and depth={self.depth_triple_blocks + self.depth_single_blocks}."
            )

        self.interleaved_audio_visual_rope = model_args.get("interleaved_audio_visual_rope", False)

        # Condition projection. Default to linear projection.
        self.condition_projection = model_args.get("condition_projection", "linear")
        self.condition_dim = model_args.get("condition_dim", None)
        self.use_attention_mask = model_args.get("use_attention_mask", False)

        self.patch_size = model_args.get("patch_size", 1)
        self.visual_in_channels = model_args.get("clip_dim", 768)
        self.audio_vae_latent_dim = model_args.get("audio_vae_latent_dim", 128)
        self.out_channels = self.audio_vae_latent_dim 
        self.unpatchify_channels = self.out_channels
        self.reverse = model_args.get("reverse", False)

        self.num_heads = model_args.get("num_heads", 24)
        self.hidden_size = model_args.get("hidden_size", 3072)
        self.rope_dim_list = model_args.get("rope_dim_list", None)
        self.mlp_ratio = model_args.get("mlp_ratio", 4.0)
        self.mlp_act_type = model_args.get("mlp_act_type", "gelu_tanh")

        self.qkv_bias = model_args.get("qkv_bias", True)
        self.qk_norm = model_args.get("qk_norm", True)
        self.qk_norm_type = model_args.get("qk_norm_type", "rms")
        self.attn_mode = model_args.get("attn_mode", "torch")

        self.embedder_type = model_args.get("embedder_type", "default")

        # sync condition things
        self.sync_modulation = model_args.get("sync_modulation", False)
        self.add_sync_feat_to_audio = model_args.get("add_sync_feat_to_audio", False)
        self.sync_feat_dim = model_args.get("sync_feat_dim", 768)
        self.sync_in_ksz = model_args.get("sync_in_ksz", 1)

        # condition tokens length
        self.clip_len = model_args.get("clip_length", 64)
        self.sync_len = model_args.get("sync_length", 192)

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(f"Hidden size {self.hidden_size} must be divisible by num_heads {self.num_heads}")

        # Build audio patchify layer and visual gated linear projection
        self.patch_size = 1
        self.audio_embedder = PatchEmbed1D(self.patch_size, self.audio_vae_latent_dim, self.hidden_size, **factory_kwargs)
        self.visual_proj = SwiGLU(self.visual_in_channels, hidden_dim=self.hidden_size, out_dim=self.hidden_size)

        # condition
        if self.condition_projection == "linear":
            self.cond_in = ConditionProjection(
                self.condition_dim, self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
        else:
            raise NotImplementedError(f"Unsupported condition_projection: {self.condition_projection}")

        # time modulation
        self.time_in = TimestepEmbedder(self.hidden_size, get_activation_layer("silu"), **factory_kwargs)

        # visual sync embedder if needed
        if self.sync_in_ksz == 1:
            sync_in_padding = 0
        elif self.sync_in_ksz == 3:
            sync_in_padding = 1
        else:
            raise ValueError
        if self.sync_modulation or self.add_sync_feat_to_audio:
            self.sync_in = nn.Sequential(
                nn.Linear(self.sync_feat_dim, self.hidden_size),
                nn.SiLU(),
                ConvMLP(self.hidden_size, self.hidden_size * 4, kernel_size=self.sync_in_ksz, padding=sync_in_padding),
            )
            self.sync_pos_emb = nn.Parameter(torch.zeros((1, 1, 8, self.sync_feat_dim)))

        self.triple_blocks = nn.ModuleList(
            [
                TwoStreamCABlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    mlp_act_type=self.mlp_act_type,
                    qk_norm=self.qk_norm,
                    qk_norm_type=self.qk_norm_type,
                    qkv_bias=self.qkv_bias,
                    attn_mode=self.attn_mode,
                    reverse=self.reverse,
                    interleaved_audio_visual_rope=self.interleaved_audio_visual_rope,
                    **factory_kwargs,
                )
                for _ in range(self.depth_triple_blocks)
            ]
        )


        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qk_norm_type=self.qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(self.depth_single_blocks)
            ]
        )

        self.final_layer = FinalLayer1D(
            self.hidden_size, self.patch_size, self.out_channels, get_activation_layer("silu"), **factory_kwargs
        )
        self.unpatchify_channels = self.out_channels

        self.empty_clip_feat = nn.Parameter(torch.zeros(1, self.visual_in_channels), requires_grad=True)
        self.empty_sync_feat = nn.Parameter(torch.zeros(1, self.sync_feat_dim), requires_grad=True)
        nn.init.constant_(self.empty_clip_feat, 0)
        nn.init.constant_(self.empty_sync_feat, 0)

    def get_empty_string_sequence(self, bs=None) -> torch.Tensor:
        if bs is None:
            return self.empty_string_feat
        else:
            return self.empty_string_feat.unsqueeze(0).expand(bs, -1, -1)

    def get_empty_clip_sequence(self, bs=None, len=None) -> torch.Tensor:
        len = len if len is not None else self.clip_len
        if bs is None:
            return self.empty_clip_feat.expand(len, -1)  # 15s
        else:
            return self.empty_clip_feat.unsqueeze(0).expand(bs, len, -1)  # 15s

    def get_empty_sync_sequence(self, bs=None, len=None) -> torch.Tensor:
        len = len if len is not None else self.sync_len
        if bs is None:
            return self.empty_sync_feat.expand(len, -1)
        else:
            return self.empty_sync_feat.unsqueeze(0).expand(bs, len, -1)

    def build_rope_for_audio_visual(self, audio_emb_len, visual_cond_len):
        assert self.patch_size == 1
        # ======================================== Build RoPE for audio tokens ======================================
        target_ndim = 1  # n-d RoPE
        rope_sizes = [audio_emb_len]
        head_dim = self.hidden_size // self.num_heads
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list=rope_dim_list,
            start=rope_sizes,
            theta=10000,
            use_real=True,
            theta_rescale_factor=1.0,
        )

        # ========================== Build RoPE for clip tokens =========================
        target_ndim = 1  # n-d RoPE
        rope_sizes = [visual_cond_len]
        head_dim = self.hidden_size // self.num_heads
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        v_freqs_cos, v_freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list=rope_dim_list,
            start=rope_sizes,
            theta=10000,
            use_real=True,
            theta_rescale_factor=1.0,
            freq_scaling=1.0 * audio_emb_len / visual_cond_len,
        )
        return freqs_cos, freqs_sin, v_freqs_cos, v_freqs_sin

    def build_rope_for_interleaved_audio_visual(self, total_len):
        assert self.patch_size == 1
        # ========================== Build RoPE for audio tokens ========================
        target_ndim = 1  # n-d RoPE
        rope_sizes = [total_len]
        head_dim = self.hidden_size // self.num_heads
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [head_dim // target_ndim for _ in range(target_ndim)]
        assert sum(rope_dim_list) == head_dim, "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list=rope_dim_list,
            start=rope_sizes,
            theta=10000,
            use_real=True,
            theta_rescale_factor=1.0,
        )
        return freqs_cos, freqs_sin

    def set_attn_mode(self, new_mode):
        for block in self.triple_blocks:
            block.set_attn_mode(new_mode)
        for block in self.single_blocks:
            block.set_attn_mode(new_mode)

    def enable_deterministic(self):
        for block in self.triple_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.triple_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        clip_feat: Optional[torch.Tensor] = None,
        cond: torch.Tensor = None,
        audio_mask: Optional[torch.Tensor] = None,
        cond_mask: torch.Tensor = None,
        sync_feat: Optional[torch.Tensor] = None,
        drop_visual: Optional[List[bool]] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = {}
        audio = x
        bs, _, ol = x.shape
        tl = ol // self.patch_size

        # Prepare learnable empty conditions for visual condition
        if drop_visual is not None:
            clip_feat[drop_visual] = self.get_empty_clip_sequence().to(dtype=clip_feat.dtype)
            sync_feat[drop_visual] = self.get_empty_sync_sequence().to(dtype=sync_feat.dtype)

        # ========================= Prepare time & visual modulation =========================
        vec = self.time_in(t)
        sync_vec = None
        if self.sync_modulation:
            assert sync_feat is not None and sync_feat.shape[1] % 8 == 0
            sync_feat = sync_feat.view(bs, int(sync_feat.shape[1] / 8), 8, self.sync_feat_dim) + self.sync_pos_emb
            sync_feat = sync_feat.view(bs, -1, self.sync_feat_dim)  # bs, num_segments * 8, channels
            sync_vec = self.sync_in(sync_feat)  # bs, num_segments * 8, c
            sync_vec = (
                F.interpolate(sync_vec.transpose(1, 2), size=(tl), mode="nearest-exact").contiguous().transpose(1, 2)
            )  # bs, tl, c
            sync_vec = sync_vec + vec.unsqueeze(1)
        elif self.add_sync_feat_to_audio:
            assert sync_feat is not None and sync_feat.shape[1] % 8 == 0
            sync_feat = sync_feat.view(bs, sync_feat.shape[1] // 8, 8, self.sync_feat_dim) + self.sync_pos_emb
            sync_feat = sync_feat.view(bs, -1, self.sync_feat_dim)  # bs, num_segments * 8, channels
            sync_feat = self.sync_in(sync_feat)  # bs, num_segments * 8, c
            add_sync_feat_to_audio = (
                F.interpolate(sync_feat.transpose(1, 2), size=(tl), mode="nearest-exact").contiguous().transpose(1, 2)
            )  # bs, tl, c

        # ========================= Get text, audio and video clip embedding =========================
        cond = self.cond_in(cond)
        cond_seq_len = cond.shape[1]

        audio = self.audio_embedder(x)
        audio_seq_len = audio.shape[1]
        v_cond = self.visual_proj(clip_feat)
        v_cond_seq_len = v_cond.shape[1]

        # ========================= Compute attention mask =========================
        attn_mask = None
        if self.use_attention_mask:
            assert cond_mask is not None
            batch_size = audio.shape[0]
            seq_len = cond_seq_len + v_cond_seq_len + audio_seq_len

            # get default audio_mask and v_cond_mask
            audio_mask = torch.ones((batch_size, audio_seq_len), dtype=torch.bool, device=audio.device)
            v_cond_mask = torch.ones((batch_size, v_cond_seq_len), dtype=torch.bool, device=audio.device)

            # batch_size x seq_len
            concat_mask = torch.cat([cond_mask, v_cond_mask, audio_mask], dim=1)
            # batch_size x 1 x seq_len x seq_len
            attn_mask_1 = concat_mask.view(batch_size, 1, 1, seq_len).repeat(1, 1, seq_len, 1)
            # batch_size x 1 x seq_len x seq_len
            attn_mask_2 = attn_mask_1.transpose(2, 3)
            # batch_size x 1 x seq_len x seq_len, 1 for broadcasting of num_heads
            attn_mask = (attn_mask_1 & attn_mask_2).bool()
            # avoids self-attention weight being NaN for text padding tokens
            attn_mask[:, :, :, 0] = True


        # ========================= Build rope for audio and clip tokens =========================
        if self.interleaved_audio_visual_rope:
            freqs_cos, freqs_sin = self.build_rope_for_interleaved_audio_visual(audio_seq_len * 2)
            v_freqs_cos = v_freqs_sin = None
        else:
            freqs_cos, freqs_sin, v_freqs_cos, v_freqs_sin = self.build_rope_for_audio_visual(
                audio_seq_len, v_cond_seq_len
            )

        # ========================= Pass through DiT blocks =========================
        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        v_freqs_cis = (v_freqs_cos, v_freqs_sin) if v_freqs_cos is not None else None

        if self.add_sync_feat_to_audio:
            add_sync_layer = 0
        assert (
            add_sync_layer < self.depth_triple_blocks
        ), f"The layer to add mel_spectrogram feature and sync feature should in the triple_stream_blocks (n: {self.depth_triple_blocks})."
        # Triple-stream blocks
        for layer_num, block in enumerate(self.triple_blocks):
            if self.add_sync_feat_to_audio and layer_num == add_sync_layer:
                audio = audio + add_sync_feat_to_audio
            triple_block_args = [audio, cond, v_cond, attn_mask, vec, freqs_cis, v_freqs_cis, sync_vec]
            if (
                self.training
                and self.gradient_checkpoint
                and (self.gradient_checkpoint_layers == -1 or layer_num < self.gradient_checkpoint_layers)
            ):
                audio, cond, v_cond = torch.utils.checkpoint.checkpoint(
                    ckpt_wrapper(block), *triple_block_args, use_reentrant=False
                )
            else:
                audio, cond, v_cond = block(*triple_block_args)

        x = audio 
        if sync_vec is not None:
            vec = vec.unsqueeze(1).repeat(1, cond_seq_len + v_cond_seq_len, 1)
            vec = torch.cat((vec, sync_vec), dim=1)

        freqs_cos, freqs_sin, _, _ = self.build_rope_for_audio_visual(audio_seq_len, v_cond_seq_len)
        if self.add_sync_feat_to_audio:
            vec = add_sync_feat_to_audio + vec.unsqueeze(dim=1)
        if len(self.single_blocks) > 0:
            for layer_num, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    (freqs_cos, freqs_sin),
                ]
                if (
                    self.training
                    and self.gradient_checkpoint
                    and (
                        self.gradient_checkpoint_layers == -1
                        or layer_num + len(self.triple_blocks) < self.gradient_checkpoint_layers
                    )
                ):
                    x = torch.utils.checkpoint.checkpoint(ckpt_wrapper(block), *single_block_args, use_reentrant=False)
                else:
                    x = block(*single_block_args)

        audio = x

        # ========================= Final layer =========================
        if sync_vec is not None:
            vec = sync_vec
        audio = self.final_layer(audio, vec)  # (N, T, patch_size * out_channels)
        audio = self.unpatchify1d(audio, tl)

        if return_dict:
            out["x"] = audio
            return out
        return audio

    def unpatchify1d(self, x, l):
        # x: (N, L, patch_size * C)
        # audio: (N, C, T), T == L * patch_size
        c = self.unpatchify_channels
        p = self.patch_size
        assert l == x.shape[1]

        x = x.reshape(shape=(x.shape[0], l, p, c))
        x = torch.einsum("ntpc->nctp", x)
        audio = x.reshape(shape=(x.shape[0], c, l * p))
        return audio

    def params_count(self):
        counts = {
            "triple": sum(
                [
                    sum(p.numel() for p in block.audio_cross_q.parameters())
                    + sum(p.numel() for p in block.v_cond_cross_q.parameters())
                    + sum(p.numel() for p in block.text_cross_kv.parameters())
                    + sum(p.numel() for p in block.audio_self_attn_qkv.parameters())
                    + sum(p.numel() for p in block.v_cond_attn_qkv.parameters())
                    + sum(p.numel() for p in block.audio_mlp.parameters())
                    + sum(p.numel() for p in block.audio_self_proj.parameters())
                    + sum(p.numel() for p in block.v_cond_self_proj.parameters())
                    + sum(p.numel() for p in block.v_cond_mlp.parameters())
                    for block in self.triple_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }

        counts["attn+mlp"] = counts["triple"] + counts["single"]
        return counts
