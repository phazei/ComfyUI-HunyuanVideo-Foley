# Modified from timm library:
# https://github.com/huggingface/pytorch-image-models/blob/648aaa41233ba83eb38faf5ba9d415d574823241/timm/layers/mlp.py#L13

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modulate_layers import modulate
from ...utils.helper import to_2tuple

class MLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_features=None,
        act_layer=nn.GELU,
        norm_layer=None,
        bias=True,
        drop=0.0,
        use_conv=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        out_features = out_features or in_channels
        hidden_channels = hidden_channels or in_channels
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_channels, hidden_channels, bias=bias[0], **factory_kwargs)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_channels, **factory_kwargs) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_channels, out_features, bias=bias[1], **factory_kwargs)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# copied from https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py
# only used when use_vanilla is True
class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True, **factory_kwargs)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True, **factory_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class LinearWarpforSingle(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias=True, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias, **factory_kwargs)

    def forward(self, x, y):
        z = torch.cat([x, y], dim=2)
        return self.fc(z)

class FinalLayer1D(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, act_layer, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        # Just use LayerNorm for the final layer
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True, **factory_kwargs)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Here we don't distinguish between the modulate types. Just use the simple one.
        self.adaLN_modulation = nn.Sequential(
            act_layer(), nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs)
        )
        # Zero-initialize the modulation
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift=shift, scale=scale)
        x = self.linear(x)
        return x


class ChannelLastConv1d(nn.Conv1d):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class ConvMLP(nn.Module):

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        kernel_size: int = 3,
        padding: int = 1,
        device=None,
        dtype=None,
    ):
        """
        Convolutional MLP module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1: Linear transformation for the first layer.
            w2: Linear transformation for the second layer.
            w3: Linear transformation for the third layer.

        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChannelLastConv1d(dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding, **factory_kwargs)
        self.w2 = ChannelLastConv1d(hidden_dim, dim, bias=False, kernel_size=kernel_size, padding=padding, **factory_kwargs)
        self.w3 = ChannelLastConv1d(dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding, **factory_kwargs)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
