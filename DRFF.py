from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SKConv(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernels: Sequence[int] = (3, 5),
        stride: int = 1,
        groups: int = 1,
        reduction: int = 16,
        min_dim: int = 32,
        norm_layer: Optional[type[nn.Module]] = nn.BatchNorm2d,
        act_layer: Optional[type[nn.Module]] = nn.ReLU,
    ) -> None:
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if not kernels or len(kernels) < 2:
            raise ValueError("kernels must contain at least 2 branch kernel sizes")
        if any(k % 2 == 0 for k in kernels):
            raise ValueError(f"All kernels should be odd to preserve spatial size, got {kernels}")

        self.channels = int(channels)
        self.kernels = tuple(int(k) for k in kernels)
        self.M = len(self.kernels)
        self.stride = int(stride)
        self.groups = int(groups)

        self.in_channels = self.channels
        self.out_channels = self.channels

        branches: List[nn.Module] = []
        for k in self.kernels:
            padding = k // 2
            layers: List[nn.Module] = [
                nn.Conv2d(
                    self.channels,
                    self.channels,
                    kernel_size=k,
                    stride=self.stride,
                    padding=padding,
                    groups=self.groups,
                    bias=False,
                )
            ]
            if norm_layer is not None:
                layers.append(norm_layer(self.channels))
            if act_layer is not None:
                layers.append(act_layer(inplace=True))
            branches.append(nn.Sequential(*layers))
        self.branches = nn.ModuleList(branches)

        d = max(self.channels // int(reduction), int(min_dim))
        self.fc = nn.Sequential(
            nn.Conv2d(self.channels, d, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.fcs = nn.ModuleList([
            nn.Conv2d(d, self.channels, kernel_size=1, bias=True) for _ in range(self.M)
        ])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]

        u = torch.stack(feats, dim=1)
        s = u.sum(dim=1)
        z = F.adaptive_avg_pool2d(s, 1)
        z = self.fc(z)

        attn = torch.stack([fc_i(z) for fc_i in self.fcs], dim=1)
        attn = self.softmax(attn)

        v = (u * attn).sum(dim=1)
        return v


class StageFeatureCache:

    def __init__(self, model: nn.Module, stage_indices: Sequence[int] = (0, 1, 2, 3)):
        self.model = model
        self.stage_indices = tuple(int(i) for i in stage_indices)
        self._features: Dict[int, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

        feature_info = getattr(model, "feature_info", None)
        all_modules = dict(model.named_modules())

        for i in self.stage_indices:
            target_module = None

            if feature_info is not None and i < len(feature_info):
                mod_name = feature_info[i]["module"]
                target_module = all_modules.get(mod_name, None)
            else:
                stages = getattr(model, "stages", None)
                if stages is not None and i < len(stages):
                    target_module = stages[i]

            if target_module is not None:
                def _make_hook(stage_id: int):
                    def _hook(_m, _inp, out):
                        _out = out[0] if isinstance(out, (list, tuple)) else out
                        if torch.is_tensor(_out):
                            if _out.dim() == 3:
                                B, L, C = _out.shape
                                H = int(math.sqrt(L))
                                _out = _out.transpose(1, 2).reshape(B, C, H, H)
                            self._features[int(stage_id)] = _out.contiguous()

                    return _hook

                self._handles.append(target_module.register_forward_hook(_make_hook(i)))
            else:
                import logging
                logging.getLogger("train").warning(f"[DRFF Cache] Stage {i} target module not found.")

    def get(self, stage_id: int) -> Optional[torch.Tensor]:
        return self._features.get(int(stage_id), None)

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()


def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] == ref.shape[-2:]:
        return x
    h, w = ref.shape[-2:]
    if x.shape[-2] < h or x.shape[-1] < w:
        return F.interpolate(x, size=(h, w), mode="nearest")
    return F.adaptive_max_pool2d(x, output_size=(h, w))


class WeightedFusion(nn.Module):

    def __init__(self, n_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.eps = float(eps)
        self.w = nn.Parameter(torch.ones(int(n_inputs), dtype=torch.float32), requires_grad=True)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        out = inputs[0] * w[0]
        for i in range(1, len(inputs)):
            out = out + inputs[i] * w[i]
        return out


class ConvBnAct(nn.Module):

    def __init__(self, in_chs: int, out_chs: int, kernel_size: int = 1, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chs)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DRFFLayer(nn.Module):

    def __init__(self, num_levels: int, channels: int, eps: float = 1e-4,
                 sk_kernels: Sequence[int] = (3, 5),
                 sk_groups: int = 1,
                 sk_reduction: int = 16,
                 sk_min_dim: int = 32):
        super().__init__()
        if num_levels < 2: raise ValueError("num_levels >= 2 required")
        self.num_levels = num_levels
        self.channels = channels
        self.eps = eps

        def build_sk_node():
            return SKConv(
                channels=channels,
                kernels=sk_kernels,
                stride=1,
                groups=sk_groups,
                reduction=sk_reduction,
                min_dim=sk_min_dim,
                norm_layer=nn.BatchNorm2d,
                act_layer=nn.SiLU
            )

        self.td_fuse = nn.ModuleList([WeightedFusion(2, eps=eps) for _ in range(num_levels - 1)])
        self.td_conv = nn.ModuleList([build_sk_node() for _ in range(num_levels - 1)])

        if num_levels > 2:
            self.out_fuse_mid = nn.ModuleList([WeightedFusion(3, eps=eps) for _ in range(num_levels - 2)])
            self.out_conv_mid = nn.ModuleList([build_sk_node() for _ in range(num_levels - 2)])
        else:
            self.out_fuse_mid = nn.ModuleList()
            self.out_conv_mid = nn.ModuleList()

        self.out_fuse_last = WeightedFusion(2, eps=eps)
        self.out_conv_last = build_sk_node()

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        td: List[torch.Tensor] = [None] * self.num_levels  # type: ignore
        td[-1] = feats[-1]

        for i in range(self.num_levels - 2, -1, -1):
            up = _resize_like(td[i + 1], feats[i])
            fused = self.td_fuse[i]([feats[i], up])
            td[i] = self.td_conv[i](fused)

        out: List[torch.Tensor] = [None] * self.num_levels  # type: ignore
        out[0] = td[0]

        for i in range(1, self.num_levels - 1):
            down = _resize_like(out[i - 1], td[i])
            if self.num_levels > 2 and 1 <= i <= self.num_levels - 2:
                fuse = self.out_fuse_mid[i - 1]([feats[i], td[i], down])
                out[i] = self.out_conv_mid[i - 1](fuse)
            else:
                out[i] = feats[i] + td[i] + down

        down_last = _resize_like(out[-2], feats[-1])
        fused_last = self.out_fuse_last([feats[-1], down_last])
        out[-1] = self.out_conv_last(fused_last)

        return out


@dataclass
class DRFFConfig:
    out_channels: Optional[int] = None
    num_layers: int = 1
    eps: float = 1e-4
    sk_kernels: Tuple[int, ...] = (3, 5)
    sk_groups: Optional[int] = None
    sk_reduction: int = 16
    sk_min_dim: int = 32


class DRFF(nn.Module):
    def __init__(self, in_channels: Optional[Sequence[int]] = None, cfg: Optional[DRFFConfig] = None):
        super().__init__()
        self.cfg = cfg or DRFFConfig()
        self._given_in_channels = tuple(int(c) for c in in_channels) if in_channels is not None else None
        self._built = False
        self.proj = nn.ModuleList()
        self.layers = nn.ModuleList()

        if self._given_in_channels is not None and self.cfg.out_channels is not None:
            self._build(self._given_in_channels, int(self.cfg.out_channels))

    def _build(self, in_channels: Sequence[int], out_channels: int) -> None:
        self.out_channels = out_channels

        self.proj = nn.ModuleList([
            ConvBnAct(c, out_channels, kernel_size=1) for c in in_channels
        ])

        groups = self.cfg.sk_groups
        if groups is None:
            groups = out_channels

        if out_channels % groups != 0:
            groups = 1

        self.layers = nn.ModuleList([
            DRFFLayer(
                num_levels=len(in_channels),
                channels=out_channels,
                eps=self.cfg.eps,
                sk_kernels=self.cfg.sk_kernels,
                sk_groups=groups,
                sk_reduction=self.cfg.sk_reduction,
                sk_min_dim=self.cfg.sk_min_dim
            )
            for _ in range(self.cfg.num_layers)
        ])
        self._built = True

    def forward(self, feats: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if not self._built:
            in_ch = [f.shape[1] for f in feats]
            out_ch = self.cfg.out_channels if self.cfg.out_channels else in_ch[-1]
            self._build(in_ch, int(out_ch))
            self.to(feats[-1].device)

        x = [p(f) for f, p in zip(feats, self.proj)]
        for layer in self.layers:
            x = layer(x)
        return x


class DRFFEnhancer(nn.Module):

    def __init__(self, cache: StageFeatureCache,
                 stage_indices: Sequence[int], target_stage: int,
                 cfg: DRFFConfig, force_match: bool = True):
        super().__init__()
        self.cache = cache
        self.stage_indices = stage_indices
        self.target_stage = target_stage
        self.bifpn = DRFF(cfg=cfg)
        self.force_match = force_match
        self._match_conv: Optional[nn.Module] = None

    def forward(self, target: torch.Tensor) -> torch.Tensor:
        was_3d = False

        if target.dim() == 3:
            was_3d = True
            B, L, C = target.shape
            H = int(math.sqrt(L))
            target = target.transpose(1, 2).reshape(B, C, H, H).contiguous()
        else:
            target = target.contiguous()

        feats = []
        b, _, h, w = target.shape
        for s in self.stage_indices:
            if s == self.target_stage:
                f = target
            else:
                f = self.cache.get(s)

            if f is None or f.shape[0] != b:
                f = target.new_zeros(b, target.shape[1], h, w)  # fallback
            feats.append(f)

        fused_list = self.bifpn(feats)

        if self.target_stage in self.stage_indices:
            idx = self.stage_indices.index(self.target_stage)
        else:
            idx = -1
        out = fused_list[idx]

        out = _resize_like(out, target)
        if self.force_match and out.shape[1] != target.shape[1]:
            if self._match_conv is None:
                self._match_conv = nn.Sequential(
                    nn.Conv2d(out.shape[1], target.shape[1], 1, bias=False),
                    nn.BatchNorm2d(target.shape[1]),
                    nn.SiLU(inplace=True)
                ).to(target.device)
            out = self._match_conv(out)

        if was_3d:
            B, C, H, W = out.shape
            out = out.flatten(2).transpose(1, 2).contiguous()

        return out