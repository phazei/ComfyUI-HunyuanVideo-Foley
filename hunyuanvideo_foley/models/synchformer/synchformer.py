import logging
import math
from typing import Any, Mapping

import einops
import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from .motionformer import MotionFormer
from .ast_model import AST
from .utils import Config


class Synchformer(nn.Module):

    def __init__(self):
        super().__init__()

        self.vfeat_extractor = MotionFormer(
            extract_features=True,
            factorize_space_time=True,
            agg_space_module="TransformerEncoderLayer",
            agg_time_module="torch.nn.Identity",
            add_global_repr=False,
        )
        self.afeat_extractor = AST(
            extract_features=True,
            max_spec_t=66,
            factorize_freq_time=True,
            agg_freq_module="TransformerEncoderLayer",
            agg_time_module="torch.nn.Identity",
            add_global_repr=False,
        )

        # # bridging the s3d latent dim (1024) into what is specified in the config
        # # to match e.g. the transformer dim
        self.vproj = nn.Linear(in_features=768, out_features=768)
        self.aproj = nn.Linear(in_features=768, out_features=768)
        self.transformer = GlobalTransformer(
            tok_pdrop=0.0, embd_pdrop=0.1, resid_pdrop=0.1, attn_pdrop=0.1, n_layer=3, n_head=8, n_embd=768
        )

    def forward(self, vis):
        B, S, Tv, C, H, W = vis.shape
        vis = vis.permute(0, 1, 3, 2, 4, 5)  # (B, S, C, Tv, H, W)
        # feat extractors return a tuple of segment-level and global features (ignored for sync)
        # (B, S, tv, D), e.g. (B, 7, 8, 768)
        vis = self.vfeat_extractor(vis)
        return vis

    def compare_v_a(self, vis: torch.Tensor, aud: torch.Tensor):
        vis = self.vproj(vis)
        aud = self.aproj(aud)

        B, S, tv, D = vis.shape
        B, S, ta, D = aud.shape
        vis = vis.view(B, S * tv, D)  # (B, S*tv, D)
        aud = aud.view(B, S * ta, D)  # (B, S*ta, D)
        # print(vis.shape, aud.shape)

        # self.transformer will concatenate the vis and aud in one sequence with aux tokens,
        # ie `CvvvvMaaaaaa`, and will return the logits for the CLS tokens
        logits = self.transformer(vis, aud)  # (B, cls); or (B, cls) and (B, 2) if DoubtingTransformer

        return logits

    def extract_vfeats(self, vis):
        B, S, Tv, C, H, W = vis.shape
        vis = vis.permute(0, 1, 3, 2, 4, 5)  # (B, S, C, Tv, H, W)
        # feat extractors return a tuple of segment-level and global features (ignored for sync)
        # (B, S, tv, D), e.g. (B, 7, 8, 768)
        vis = self.vfeat_extractor(vis)
        return vis

    def extract_afeats(self, aud):
        B, S, _, Fa, Ta = aud.shape
        aud = aud.view(B, S, Fa, Ta).permute(0, 1, 3, 2)  # (B, S, Ta, F)
        # (B, S, ta, D), e.g. (B, 7, 6, 768)
        aud, _ = self.afeat_extractor(aud)
        return aud

    def compute_loss(self, logits, targets, loss_fn: str = None):
        loss = None
        if targets is not None:
            if loss_fn is None or loss_fn == "cross_entropy":
                # logits: (B, cls) and targets: (B,)
                loss = F.cross_entropy(logits, targets)
            else:
                raise NotImplementedError(f"Loss {loss_fn} not implemented")
        return loss

    def load_state_dict(self, sd: Mapping[str, Any], strict: bool = True):
        # discard all entries except vfeat_extractor
        # sd = {k: v for k, v in sd.items() if k.startswith('vfeat_extractor')}

        return super().load_state_dict(sd, strict)


class RandInitPositionalEncoding(nn.Module):
    """Random inited trainable pos embedding. It is just applied on the sequence, thus respects no priors."""

    def __init__(self, block_shape: list, n_embd: int):
        super().__init__()
        self.block_shape = block_shape
        self.n_embd = n_embd
        self.pos_emb = nn.Parameter(torch.randn(1, *block_shape, n_embd))

    def forward(self, token_embeddings):
        return token_embeddings + self.pos_emb


class GlobalTransformer(torch.nn.Module):
    """Same as in SparseSync but without the selector transformers and the head"""

    def __init__(
        self,
        tok_pdrop=0.0,
        embd_pdrop=0.1,
        resid_pdrop=0.1,
        attn_pdrop=0.1,
        n_layer=3,
        n_head=8,
        n_embd=768,
        pos_emb_block_shape=[
            198,
        ],
        n_off_head_out=21,
    ) -> None:
        super().__init__()
        self.config = Config(
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
        )
        # input norm
        self.vis_in_lnorm = torch.nn.LayerNorm(n_embd)
        self.aud_in_lnorm = torch.nn.LayerNorm(n_embd)
        # aux tokens
        self.OFF_tok = torch.nn.Parameter(torch.randn(1, 1, n_embd))
        self.MOD_tok = torch.nn.Parameter(torch.randn(1, 1, n_embd))
        # whole token dropout
        self.tok_pdrop = tok_pdrop
        self.tok_drop_vis = torch.nn.Dropout1d(tok_pdrop)
        self.tok_drop_aud = torch.nn.Dropout1d(tok_pdrop)
        # maybe add pos emb
        self.pos_emb_cfg = RandInitPositionalEncoding(
            block_shape=pos_emb_block_shape,
            n_embd=n_embd,
        )
        # the stem
        self.drop = torch.nn.Dropout(embd_pdrop)
        self.blocks = torch.nn.Sequential(*[Block(self.config) for _ in range(n_layer)])
        # pre-output norm
        self.ln_f = torch.nn.LayerNorm(n_embd)
        # maybe add a head
        self.off_head = torch.nn.Linear(in_features=n_embd, out_features=n_off_head_out)

    def forward(self, v: torch.Tensor, a: torch.Tensor, targets=None, attempt_to_apply_heads=True):
        B, Sv, D = v.shape
        B, Sa, D = a.shape
        # broadcasting special tokens to the batch size
        off_tok = einops.repeat(self.OFF_tok, "1 1 d -> b 1 d", b=B)
        mod_tok = einops.repeat(self.MOD_tok, "1 1 d -> b 1 d", b=B)
        # norm
        v, a = self.vis_in_lnorm(v), self.aud_in_lnorm(a)
        # maybe whole token dropout
        if self.tok_pdrop > 0:
            v, a = self.tok_drop_vis(v), self.tok_drop_aud(a)
        # (B, 1+Sv+1+Sa, D)
        x = torch.cat((off_tok, v, mod_tok, a), dim=1)
        # maybe add pos emb
        if hasattr(self, "pos_emb_cfg"):
            x = self.pos_emb_cfg(x)
        # dropout -> stem -> norm
        x = self.drop(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        # maybe add heads
        if attempt_to_apply_heads and hasattr(self, "off_head"):
            x = self.off_head(x[:, 0, :])
        return x


class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # # causal mask to ensure that attention is only applied to the left in the input sequence
        # mask = torch.tril(torch.ones(config.block_size,
        #                              config.block_size))
        # if hasattr(config, "n_unmasked"):
        #     mask[:config.n_unmasked, :config.n_unmasked] = 1
        # self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = self.attn_drop(att) @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))

        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


def make_class_grid(
    leftmost_val,
    rightmost_val,
    grid_size,
    add_extreme_offset: bool = False,
    seg_size_vframes: int = None,
    nseg: int = None,
    step_size_seg: float = None,
    vfps: float = None,
):
    assert grid_size >= 3, f"grid_size: {grid_size} doesnot make sense. If =2 -> (-1,1); =1 -> (-1); =0 -> ()"
    grid = torch.from_numpy(np.linspace(leftmost_val, rightmost_val, grid_size)).float()
    if add_extreme_offset:
        assert all([seg_size_vframes, nseg, step_size_seg]), f"{seg_size_vframes} {nseg} {step_size_seg}"
        seg_size_sec = seg_size_vframes / vfps
        trim_size_in_seg = nseg - (1 - step_size_seg) * (nseg - 1)
        extreme_value = trim_size_in_seg * seg_size_sec
        grid = torch.cat([grid, torch.tensor([extreme_value])])  # adding extreme offset to the class grid
    return grid


# from synchformer
def pad_or_truncate(audio: torch.Tensor, max_spec_t: int, pad_mode: str = "constant", pad_value: float = 0.0):
    difference = max_spec_t - audio.shape[-1]  # safe for batched input
    # pad or truncate, depending on difference
    if difference > 0:
        # pad the last dim (time) -> (..., n_mels, 0+time+difference)  # safe for batched input
        pad_dims = (0, difference)
        audio = torch.nn.functional.pad(audio, pad_dims, pad_mode, pad_value)
    elif difference < 0:
        print(f"Truncating spec ({audio.shape}) to max_spec_t ({max_spec_t}).")
        audio = audio[..., :max_spec_t]  # safe for batched input
    return audio


def encode_audio_with_sync(
    synchformer: Synchformer, x: torch.Tensor, mel: torchaudio.transforms.MelSpectrogram
) -> torch.Tensor:
    b, t = x.shape

    # partition the video
    segment_size = 10240
    step_size = 10240 // 2
    num_segments = (t - segment_size) // step_size + 1
    segments = []
    for i in range(num_segments):
        segments.append(x[:, i * step_size : i * step_size + segment_size])
    x = torch.stack(segments, dim=1)  # (B, S, T, C, H, W)

    x = mel(x)
    x = torch.log(x + 1e-6)
    x = pad_or_truncate(x, 66)

    mean = -4.2677393
    std = 4.5689974
    x = (x - mean) / (2 * std)
    # x: B * S * 128 * 66
    x = synchformer.extract_afeats(x.unsqueeze(2))
    return x


def read_audio(filename, expected_length=int(16000 * 4)):
    waveform, sr = torchaudio.load(filename)
    waveform = waveform.mean(dim=0)

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler[sr](waveform)

    waveform = waveform[:expected_length]
    if waveform.shape[0] != expected_length:
        raise ValueError(f"Audio {filename} is too short")

    waveform = waveform.squeeze()

    return waveform


if __name__ == "__main__":
    synchformer = Synchformer().cuda().eval()

    # mmaudio provided synchformer ckpt
    synchformer.load_state_dict(
        torch.load(
            os.environ.get("SYNCHFORMER_WEIGHTS", f"weights/synchformer.pth"),
            weights_only=True,
            map_location="cpu",
        )
    )

    sync_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        win_length=400,
        hop_length=160,
        n_fft=1024,
        n_mels=128,
    )
