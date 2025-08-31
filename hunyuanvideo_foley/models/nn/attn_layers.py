import importlib.metadata
import math
from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


try:
    from torch import compiler as _compiler
    _disable_compile = _compiler.disable
except Exception:
    import torch._dynamo as _dynamo
    _disable_compile = _dynamo.disable

try:
    from sageattention import sageattn, sageattn_varlen
except Exception:
    sageattn = None
    sageattn_varlen = None

try:
    from flash_attn import (
        flash_attn_qkvpacked_func,
        flash_attn_kvpacked_func,
        flash_attn_varlen_kvpacked_func,
        flash_attn_varlen_qkvpacked_func,
    )
    from flash_attn.bert_padding import pad_input, unpad_input
except ImportError:
    flash_attn_qkvpacked_func = flash_attn_kvpacked_func = flash_attn_varlen_kvpacked_func = None
    index_first_axis = pad_input = unpad_input = None
from packaging import version
from transformers.utils.import_utils import _is_package_available

from .norm_layers import get_norm_layer


_DEBUG_FLASH = False

@_disable_compile()  # ensure this never gets traced/compiled
def _flash_log(tag: str, **tensors):
    if not _DEBUG_FLASH:
        return
    lines = [f"[HY-FOLEY/ATTN] {tag}"]
    for name, t in tensors.items():
        if isinstance(t, torch.Tensor):
            try:
                lines.append(f"  {name}: dev={t.device}, dtype={t.dtype}, shape={tuple(t.shape)}")
            except Exception:
                lines.append(f"  {name}: <tensor>")
        else:
            lines.append(f"  {name}: {type(t).__name__}")
    print("\n".join(lines))


@_disable_compile()
def _sage_call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool) -> torch.Tensor:
    """
    Safe call into sageattention.sageattn, excluded from torch.compile.
    Expects q,k,v in HND layout: [B, H, N, D]
    Coerces to fp16/bf16 for the kernel and restores original dtype on return.
    """
    assert sageattn is not None, "SageAttention not installed"

    orig_dtype = q.dtype
    # Kernel requires fp16/bf16 and consistent dtypes
    if orig_dtype not in (torch.float16, torch.bfloat16) or (q.dtype != k.dtype or q.dtype != v.dtype):
        # prefer sticking with the incoming half if it already is one, else choose fp16
        target = q.dtype if q.dtype in (torch.float16, torch.bfloat16) else torch.float16
        q = q.to(dtype=target, device=q.device, non_blocking=True)
        k = k.to(dtype=target, device=q.device, non_blocking=True)
        v = v.to(dtype=target, device=q.device, non_blocking=True)

    out = sageattn(q, k, v, tensor_layout="HND", is_causal=causal)
    # hand back the original dtype to keep surrounding math stable
    if out.dtype != orig_dtype:
        out = out.to(dtype=orig_dtype)
    return out

@_disable_compile()
def _sage_varlen_call(
    q_flat: torch.Tensor,             # [sum Nq, H, D]  (HND flattened on N)
    kv_flat: torch.Tensor,            # [sum Nk, 2, H, D]
    cu_q: torch.Tensor,               # int32 prefix sums, device=CUDA
    cu_k: torch.Tensor,               # int32 prefix sums, device=CUDA
    max_q: int,
    max_k: int,
    *, causal: bool, out_dtype: torch.dtype
) -> torch.Tensor:
    """
    Safe call into sageattention varlen kernel; excluded from torch.compile.
    Returns [sum Nq, H, D] (same layout as q_flat); caller reshapes.
    """
    assert sageattn_varlen is not None, "sageattn_varlen not available"

    # Kernel requires fp16/bf16
    if q_flat.dtype not in (torch.float16, torch.bfloat16):
        q_flat  = q_flat.to(torch.float16)
        kv_flat = kv_flat.to(torch.float16)

    out = sageattn_varlen(
        q_flat, kv_flat,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=max_q, max_seqlen_k=max_k,
        tensor_layout="HND", is_causal=causal
    )
    if out.dtype != out_dtype:
        out = out.to(out_dtype)
    return out


def reshape_for_broadcast(freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]], x: torch.Tensor, head_first=False):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Notes:
        When using FlashMHAModified, head_first should be False.
        When using Attention, head_first should be True.

    Args:
        freqs_cis (Union[torch.Tensor, Tuple[torch.Tensor]]): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim

    if isinstance(freqs_cis, tuple):
        # freqs_cis: (cos, sin) in real space
        if head_first:
            assert freqs_cis[0].shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis[0].shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis[0].shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis[0].view(*shape), freqs_cis[1].view(*shape)
    else:
        # freqs_cis: values in complex space
        if head_first:
            assert freqs_cis.shape == (
                x.shape[-2],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == ndim - 2 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        else:
            assert freqs_cis.shape == (
                x.shape[1],
                x.shape[-1],
            ), f"freqs_cis shape {freqs_cis.shape} does not match x shape {x.shape}"
            shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)


def rotate_half(x):
    x_real, x_imag = x.float().reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
    return torch.stack([-x_imag, x_real], dim=-1).flatten(3)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    head_first: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. [B, S, H, D]
        xk (torch.Tensor): Key tensor to apply rotary embeddings.   [B, S, H, D]
        freqs_cis (torch.Tensor or tuple): Precomputed frequency tensor for complex exponential.
        head_first (bool): head dimension first (except batch dim) or not.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
    xk_out = None
    if isinstance(freqs_cis, tuple):
        cos, sin = reshape_for_broadcast(freqs_cis, xq, head_first)  # [S, D]
        cos, sin = cos.to(xq.device), sin.to(xq.device)
        # real * cos - imag * sin
        # imag * cos + real * sin
        xq_out = (xq.float() * cos + rotate_half(xq.float()) * sin).type_as(xq)
        xk_out = (xk.float() * cos + rotate_half(xk.float()) * sin).type_as(xk)
    else:
        # view_as_complex will pack [..., D/2, 2](real) to [..., D/2](complex)
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [B, S, H, D//2]
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_, head_first).to(xq.device)  # [S, D//2] --> [1, S, 1, D//2]
        # (real, imag) * (cos, sin) = (real * cos - imag * sin, imag * cos + real * sin)
        # view_as_real will expand [..., D/2](complex) to [..., D/2, 2](real)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq)
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))  # [B, S, H, D//2]
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

    return xq_out, xk_out


class BasicAttentionLayer(nn.Module):
    def __init__(self, attn_mode="flash", deterministic=False):
        super().__init__()
        self.attn_mode = attn_mode
        self.deterministic = deterministic

    def set_attn_mode(self, new_mode):
        self.attn_mode = new_mode

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False


MEMORY_LAYOUT = {
    "self_flash": (
        lambda x: x,
        lambda x: x,
    ),
    "cross_flash": (
        lambda x: x,
        lambda x: x,
    ),
    "flash_torch_sp": (
        lambda x: x,
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "sage": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2)
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}


# Copyed from https://github.com/huggingface/transformers/blob/b873234cb649a24865021f0d598627ce2b24d34a/src/transformers/modeling_flash_attention_utils.py#L33C1-L57C6
def _get_unpad_data(attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Retrieves indexing data required to repad unpadded (ragged) tensors.

    Arguments:
        attention_mask (`torch.Tensor`):
            Boolean or int tensor of shape (batch_size, sequence_length), 1 means valid and 0 means not valid.

    Return:
        indices (`torch.Tensor):
            The indices of non-masked tokens from the flattened input sequence.
        cu_seqlens (`torch.Tensor`):
            The cumulative sequence lengths, used to index into ragged (unpadded) tensors. `cu_seqlens` shape is (batch_size + 1,).
        max_seqlen_in_batch (`int`):
            Maximum sequence length in batch.
    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copyed from https://github.com/huggingface/transformers/blob/b873234cb649a24865021f0d598627ce2b24d34a/src/transformers/utils/import_utils.py#L822
def is_flash_attn_greater_or_equal(library_version: str):
    if not _is_package_available("flash_attn"):
        return False

    return version.parse(importlib.metadata.version("flash_attn")) >= version.parse(library_version)


def get_kv_seqlens_with_mask(attn_mask, k, v):
    """
    Build varlen KV from a padding mask without depending on FlashAttention.
    Expects:
      attn_mask: [B, S_k] boolean (True = keep)
      k, v     : [B, S_k, A, H]
    Returns:
      cu_seqlens_k: [B+1] int32 (CUDA)
      max_seqlen_k: int
      kv_flat    : [sum S_k_valid, 2, A, H]
    """
    # --- move mask first, so _get_unpad_data runs on CUDA (no CPU subgraph) ---
    if attn_mask.device != k.device:
        _flash_log("moving attn_mask to CUDA for varlen mask ops",
                   attn_mask=attn_mask, k=k)
        attn_mask = attn_mask.to(device=k.device)

    # indices, cu sums, max len — now fully on CUDA
    indices_k, cu_seqlens_k, max_seqlen_k = _get_unpad_data(attn_mask)

    b, s1, a, d = k.shape

    # ensure devices/dtypes expected by kernels
    indices_k = indices_k.to(device=k.device, dtype=torch.long)
    if cu_seqlens_k.device != k.device:
        cu_seqlens_k = cu_seqlens_k.to(device=k.device)
    if cu_seqlens_k.dtype != torch.int32:
        cu_seqlens_k = cu_seqlens_k.to(dtype=torch.int32)

    # gather valid rows
    k = torch.index_select(k.reshape(b * s1, a, d), 0, indices_k)
    v = torch.index_select(v.reshape(b * s1, a, d), 0, indices_k)
    kv = torch.stack([k, v], dim=1)  # [sum S_k_valid, 2, A, H]

    _flash_log("built KV varlen (masked)",
               cu_seqlens_k=cu_seqlens_k, kv=kv)
    return cu_seqlens_k, max_seqlen_k, kv


def get_q_seqlens(q):
    bs, s, a, d = q.shape
    cu_seqlens_q = torch.arange(0, (bs + 1) * s, step=s, dtype=torch.int32, device=q.device)
    q = q.reshape(bs * s, a, d)
    return cu_seqlens_q, s, q

def flash_attn_no_pad(
    qkv, key_padding_mask, causal=False, dropout_p=0.0, softmax_scale=None
):
    # adapted from https://github.com/Dao-AILab/flash-attention/blob/13403e81157ba37ca525890f2f0f2137edf75311/flash_attn/flash_attention.py#L27
    batch_size = qkv.shape[0]
    seqlen = qkv.shape[1]
    nheads = qkv.shape[-2]
    x = rearrange(qkv, "b s three h d -> b s (three h d)")
    # x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch
    # x_unpad, indices, cu_seqlens, max_s
    unpad_results = unpad_input(
        x, key_padding_mask
    )

    if len(unpad_results) == 4:
        x_unpad, indices, cu_seqlens, max_s = unpad_results
    elif len(unpad_results) == 5:
        x_unpad, indices, cu_seqlens, max_s, used_seqlens_in_batch = unpad_results
    else:
        raise ValueError

    x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
    output_unpad = flash_attn_varlen_qkvpacked_func(
        x_unpad,
        cu_seqlens,
        max_s,
        dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
    )
    output = rearrange(
        pad_input(
            rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen
        ),
        "b s (h d) -> b s h d",
        h=nheads,
    )
    return output


def attention(
    q,
    k,
    v,
    mode,
    drop_rate=0,
    attn_mask=None,
    cond_mask=None,
    causal=False,
    deterministic=False,
    cu_seqlens=None,
    max_seqlen=None,
    cu_seqlens_k=None,
    max_seqlen_k=None,
    img_seq_len=None,
):
    """
    Perform QKV self attention.

    Args:
        q (torch.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads.
        k (torch.Tensor): Key tensor with shape [b, s1, a, d]
        v (torch.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Choose from 'self_flash', 'cross_flash', 'torch', and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (torch.Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, a, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        deterministic (bool): Whether to use deterministic attention. (default: False)
        cu_seqlens (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        max_seqlen (int): The maximum sequence length in the batch of q.
        cu_seqlens_k (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_k (int): The maximum sequence length in the batch of k and v.

    Returns:
        torch.Tensor: Output tensor after self attention with shape [b, s, ad]
    """
    
    _flash_log(f"path={mode}")

    if mode in ["torch", "vanilla", "self_flash", "cross_flash", "sage"]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=1)
        if isinstance(k, tuple):
            k = torch.cat(k, dim=1)
        if isinstance(v, tuple):
            v = torch.cat(v, dim=1)
        pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
        q = pre_attn_layout(q)
        k = pre_attn_layout(k)
        v = pre_attn_layout(v)


    orig_dtype = q.dtype  # remember to cast back for the rest of the model
    need_half = mode in {"self_flash", "cross_flash", "sage"}
    if need_half and q.dtype not in (torch.float16, torch.bfloat16):
        # Flash/Sage kernels require fp16/bf16 inputs
        target = torch.float16  # or torch.bfloat16
        q = q.to(target)
        k = k.to(target)
        v = v.to(target)


    if "flash" in mode:
        assert (
            flash_attn_qkvpacked_func is not None
        ), "Flash attention is not available. Please install flash_attn first."
        flash_kwargs = dict(dropout_p=drop_rate, causal=causal)
        if deterministic:
            if not is_flash_attn_greater_or_equal("2.4.1"):
                raise ValueError(
                    "Flash attention deterministic mode requires flash_attn>=2.4.1. " "Please upgrade flash_attn"
                )
            flash_kwargs["deterministic"] = deterministic

        # If someone passed mismatched lengths under 'self', treat it as cross.
        if mode == "self_flash" and q.shape[1] != k.shape[1]:
            mode = "cross_flash"

        # ---------------- SELF-ATTN ----------------
        if mode == "self_flash":
            if attn_mask is not None:
                raise ValueError("Self attention does not support attention mask")
            # q,k,v same length -> qkvpacked
            qkv = torch.stack([q, k, v], dim=2)            # [B, N, 3, A, H]
            x = flash_attn_qkvpacked_func(qkv, **flash_kwargs)

        # ------------------------------------------------------------
        # CROSS-ATTN: allow different q_len (audio/v-cond) vs kv_len (text)
        # Use varlen kernel when lengths differ; otherwise use kv-packed.
        # ------------------------------------------------------------
        elif mode == "cross_flash":
            B, Nq, A, H = q.shape
            Nk = k.shape[1]

            # Fast path: no mask & equal lengths -> kvpacked
            if attn_mask is None and Nq == Nk:
                kv = torch.stack([k, v], dim=2)            # [B, N, 2, A, H]
                x = flash_attn_kvpacked_func(q, kv, **flash_kwargs)

            else:
                # varlen path (works whether mask is present or lengths differ)
                if flash_attn_varlen_kvpacked_func is None:
                    # Last-resort fallback for rare environments: SDPA for this op only
                    # (keeps Flash on self-attn elsewhere)
                    q_ = q.transpose(1, 2).reshape(B * A, Nq, H)   # [B*A, Nq, H]
                    k_ = k.transpose(1, 2).reshape(B * A, Nk, H)
                    v_ = v.transpose(1, 2).reshape(B * A, Nk, H)
                    x_ = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=None, is_causal=causal)
                    x  = x_.reshape(B, A, Nq, H).transpose(1, 2)
                else:
                    # Build cu_seqlens for varlen (int32 on CUDA). If there is a real attn_mask,
                    # reuse the existing helpers to compute seqlens; otherwise synthesize uniform seqlens.
                    if attn_mask is not None:
                        # Use your helpers for masked varlen
                        b, s, a, h = q.shape
                        cu_seqlens_q, max_seqlen_q, q_flat = get_q_seqlens(q)
                        cu_seqlens_k, max_seqlen_k, kv_flat = get_kv_seqlens_with_mask(attn_mask, k, v)
                        out = flash_attn_varlen_kvpacked_func(
                            q_flat,
                            kv_flat,
                            cu_seqlens_q=cu_seqlens_q,
                            cu_seqlens_k=cu_seqlens_k,
                            max_seqlen_q=max_seqlen_q,
                            max_seqlen_k=max_seqlen_k,
                            **flash_kwargs,
                        )
                        x = out.reshape(b, s, a, h)
                    else:
                        # Uniform-varlen (no mask) when Nq != Nk
                        q_flat  = q.reshape(B * Nq, A, H)          # [sum Nq, A, H]
                        kv      = torch.stack([k, v], dim=2)       # [B, Nk, 2, A, H]
                        kv_flat = kv.reshape(B * Nk, 2, A, H)      # [sum Nk, 2, A, H]
                        cu_q = torch.arange(0, (B + 1) * Nq, step=Nq, dtype=torch.int32, device=q.device)
                        cu_k = torch.arange(0, (B + 1) * Nk, step=Nk, dtype=torch.int32, device=q.device)
                        out = flash_attn_varlen_kvpacked_func(
                            q_flat,
                            kv_flat,
                            cu_seqlens_q=cu_q,
                            cu_seqlens_k=cu_k,
                            max_seqlen_q=Nq,
                            max_seqlen_k=Nk,
                            **flash_kwargs,
                        )
                        x = out.view(B, Nq, A, H)

    elif mode == 'torch':
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal)

    elif mode == "sage":
        # q,k,v are [B,H,N,D] here thanks to pre_attn_layout
        if attn_mask is None:
            x = _sage_call(q, k, v, causal=causal)

        else:
            # flip to [B,S,H,D] for varlen helpers
            q_t = q.transpose(1, 2)   # [B,S,H,D]
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)

            # build varlen inputs from your existing helpers
            cu_q, max_q, q_flat = get_q_seqlens(q_t)                      # q_flat: [sum Nq, H, D] (HND-flattened)
            cu_k, max_k, kv_flat = get_kv_seqlens_with_mask(attn_mask, k_t, v_t)  # kv_flat: [sum Nk, 2, H, D]

            out_unflat = _sage_varlen_call(
                q_flat, kv_flat, cu_q, cu_k, max_q, max_k,
                causal=causal, out_dtype=q.dtype
            )
            # back to [B,H,S,D] so post_attn_layout can restore [B,S,H,D]
            b, s, a, d = q_t.shape
            x = out_unflat.reshape(b, s, a, d).transpose(1, 2)


    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.size(-1))

        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)
        if causal:
            # Only applied to self attention
            assert attn_mask is None, "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(diagonal=0)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attn_mask

        # TODO(jarvizhang): Maybe force q and k to be float32 to avoid numerical overflow
        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")


    if need_half and x.dtype != orig_dtype:
        x = x.to(orig_dtype)

    if mode in ["torch", "vanilla", "self_flash", "cross_flash", "sage"]:
        x = post_attn_layout(x).contiguous()
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out


class SelfAttentionLayer(BasicAttentionLayer):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_norm=True,
        attn_drop=0,
        proj_drop=0,
        dtype=None,
        device=None,
        norm_type="layer",
        attn_mode="self_flash",
        deterministic=False,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(attn_mode, deterministic)
        self.dim = dim
        self.num_heads = num_heads
        assert self.dim % num_heads == 0, "dim must be divisible by num_heads"
        self.head_dim = self.dim // num_heads
        self.attn_drop = attn_drop

        # This assertion is aligned with flash attention
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(dim, dim * 3, bias=qkv_bias, **factory_kwargs)

        norm_layer = get_norm_layer(norm_type)
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.out_proj = nn.Linear(dim, dim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, freqs_cis=None, attn_mask=None):
        """
        Args:
            x (torch.Tensor): (batch, seq_len, hidden_dim) (where hidden_dim = num heads * head dim)
            freqs_cis (torch.Tensor, optional): (batch, hidden_dim // 2), RoPE for image
            attn_mask (torch.Tensor, optional): (batch, seq_len, seq_len), mask for attention
        """
        b, s, d = x.shape

        # Apply QKV projection
        qkv = self.Wqkv(x)
        qkv = qkv.view(b, s, 3, self.num_heads, self.head_dim)  # [b, s, 3, a, d]
        q, k, v = qkv.unbind(dim=2)  # [b, s, a, d]

        # Apply QK-Norm if needed
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE if needed
        if freqs_cis is not None:
            qq, kk = apply_rotary_emb(q, k, freqs_cis)
            assert (
                qq.shape == q.shape and kk.shape == k.shape
            ), f"qq: {qq.shape}, q: {q.shape}, kk: {kk.shape}, k: {k.shape}"
            q, k = qq, kk

        # Apply self attention
        context = attention(
            q,
            k,
            v,
            drop_rate=self.attn_drop if self.training else 0,
            attn_mask=attn_mask,
            mode=self.attn_mode,
            deterministic=self.deterministic,
        )
        out = self.out_proj(context)
        out = self.proj_drop(out)

        return out


class CrossAttentionLayer(BasicAttentionLayer):
    def __init__(
        self,
        qdim,
        kdim,
        num_heads,
        qkv_bias=True,
        qk_norm=True,
        attn_drop=0,
        proj_drop=0,
        dtype=None,
        device=None,
        norm_type="layer",
        attn_mode="cross_flash",
        deterministic=False,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(attn_mode, deterministic)
        self.qdim = qdim
        self.kdim = kdim
        self.num_heads = num_heads
        assert self.qdim % num_heads == 0, "qdim must be divisible by num_heads"
        self.head_dim = self.qdim // num_heads
        self.attn_drop = attn_drop

        # This assertion is aligned with flash attention
        assert self.head_dim % 8 == 0 and self.head_dim <= 128, "Only support head_dim <= 128 and divisible by 8"

        self.q_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.kv_proj = nn.Linear(kdim, 2 * qdim, bias=qkv_bias, **factory_kwargs)

        norm_layer = get_norm_layer(norm_type)
        self.q_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )
        self.k_norm = (
            norm_layer(self.head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs) if qk_norm else nn.Identity()
        )

        self.out_proj = nn.Linear(qdim, qdim, bias=qkv_bias, **factory_kwargs)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y, attn_mask=None):
        """
        Args:
            x (torch.Tensor): (batch, seq_len, hidden_dim) (where hidden_dim = num heads * head dim)
            y (torch.Tensor): (batch, seq_len1, hidden_dim1)
            attn_mask (torch.Tensor): (batch, seq_len1), mask for attention
        """
        b, s, d = x.shape
        _, s1, d1 = y.shape

        q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)
        kv = self.kv_proj(y).view(b, s1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)

        # Apply QK-Norm if needed
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply cross attention
        context = attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            drop_rate=self.attn_drop if self.training else 0,
            mode=self.attn_mode,
            deterministic=self.deterministic,
        )
        out = self.out_proj(context)
        out = self.proj_drop(out)

        return out
