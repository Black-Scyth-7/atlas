"""MatchboxNet — 1D time-channel separable conv ResNet for keyword spotting.

From-scratch PyTorch implementation of the NVIDIA MatchboxNet architecture
(Majumdar & Ginsburg, 2020, arXiv:2004.08531), no NeMo dependency. Configured as
MatchboxNet-BxRxC and used here as a binary "Atlas" / "not-Atlas" classifier.

Input:  (batch, N_MFCC=64, frames)   normalized MFCC from features.py
Output: (batch, num_classes) logits  (global-average-pooled over time)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TCSConv1d(nn.Module):
    """Time-channel separable 1D conv: depthwise conv over time + pointwise mix.

    The efficiency trick behind MatchboxNet: a full (in x out x k) conv is split
    into a depthwise (in x 1 x k, grouped) conv over the time axis plus a
    pointwise (in x out x 1) conv that mixes channels — far fewer parameters.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel, padding=pad,
                                   dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class MatchboxBlock(nn.Module):
    """A MatchboxNet block: `repeat` sub-blocks + an optional 1x1 residual.

    Each sub-block is [conv -> BN -> ReLU -> Dropout]; the final sub-block holds
    its activation until after the residual is added (standard Jasper/QuartzNet
    residual placement).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, repeat: int = 1,
                 dilation: int = 1, dropout: float = 0.0,
                 residual: bool = True, separable: bool = True):
        super().__init__()
        self.residual = residual
        layers: list[nn.Module] = []
        c_in = in_ch
        for i in range(repeat):
            last = i == repeat - 1
            if separable:
                layers.append(TCSConv1d(c_in, out_ch, kernel, dilation))
            else:
                pad = (kernel // 2) * dilation
                layers.append(nn.Conv1d(c_in, out_ch, kernel, padding=pad,
                                        dilation=dilation, bias=False))
            layers.append(nn.BatchNorm1d(out_ch))
            if not last:                       # inner sub-blocks: full activation
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
            c_in = out_ch
        self.body = nn.Sequential(*layers)     # ends at BN (no final activation)

        if residual:
            self.res = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        self.out_act = nn.Sequential(nn.ReLU(inplace=True), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.body(x)
        if self.residual:
            y = y + self.res(x)
        return self.out_act(y)


class MatchboxNet(nn.Module):
    """MatchboxNet-BxRxC classifier.

    B = number of residual blocks, R = sub-blocks per block, C = block channels.
    Defaults reproduce MatchboxNet-3x2x64 (prologue/epilogue 128 channels).
    """

    def __init__(self, n_mfcc: int = 64, num_classes: int = 2,
                 b: int = 3, r: int = 2, c: int = 64,
                 kernels: tuple[int, ...] = (13, 15, 17),
                 prologue_ch: int = 128, epilogue_ch: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        if len(kernels) < b:                   # extend kernels if B > len given
            kernels = tuple(kernels) + tuple(
                kernels[-1] + 2 * (i + 1) for i in range(b - len(kernels)))

        blocks: list[nn.Module] = []
        # Prologue (B0): separable conv, no residual.
        blocks.append(MatchboxBlock(n_mfcc, prologue_ch, kernel=11, repeat=1,
                                    residual=False, dropout=dropout))
        # B residual blocks of R sub-blocks, C channels.
        c_in = prologue_ch
        for i in range(b):
            blocks.append(MatchboxBlock(c_in, c, kernel=kernels[i], repeat=r,
                                        residual=True, dropout=dropout))
            c_in = c
        # Epilogue B1: dilated separable conv.
        blocks.append(MatchboxBlock(c_in, epilogue_ch, kernel=29, dilation=2,
                                    repeat=1, residual=False, dropout=dropout))
        # Epilogue B2: pointwise (1x1) conv.
        blocks.append(MatchboxBlock(epilogue_ch, epilogue_ch, kernel=1, repeat=1,
                                    residual=False, separable=False, dropout=dropout))
        self.encoder = nn.Sequential(*blocks)

        # Decoder: pointwise conv to class logits, then global average pool.
        self.decoder = nn.Conv1d(epilogue_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_mfcc, frames)
        x = self.encoder(x)
        x = self.decoder(x)                    # (batch, num_classes, frames)
        return x.mean(dim=-1)                  # global avg pool -> (batch, num_classes)


def build(num_classes: int = 2, **kw) -> MatchboxNet:
    """MatchboxNet-3x2x64 by default; pass b/r/c to resize."""
    return MatchboxNet(num_classes=num_classes, **kw)


if __name__ == "__main__":
    from features import N_MFCC, N_FRAMES

    net = build()
    n_params = sum(p.numel() for p in net.parameters())
    x = torch.randn(2, N_MFCC, N_FRAMES)
    y = net(x)
    print(f"MatchboxNet-3x2x64  params={n_params:,}")
    print(f"input {tuple(x.shape)} -> output {tuple(y.shape)}  (expected (2, 2))")
    # Variable-length tolerance (global pool): a shorter clip still works.
    y2 = net(torch.randn(1, N_MFCC, 90))
    print(f"variable-length input (1, 64, 90) -> {tuple(y2.shape)}")
