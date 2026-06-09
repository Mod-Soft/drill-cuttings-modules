import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class GatedFusion(nn.Module):

    def __init__(self, channels: int, gate_from: str = "x"):
        super().__init__()
        assert gate_from in ("x", "xy")
        self.gate_from = gate_from

        in_ch = channels if gate_from == "x" else channels * 2
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.gate_from == "x":
            gate = self.gate(x)
        else:
            gate = self.gate(torch.cat([x, y], dim=1))
        return gate * x + (1.0 - gate) * y


class PGF(nn.Module):

    def __init__(
            self,
            dim: int,
            ssmdims: int = None,
            mlp_ratio: float = 4.0,
            drop: float = 0.0,
            drop_path: float = 0.0,
            act_layer=nn.ReLU6,
            norm_layer=nn.BatchNorm2d,
            gate_from: str = "x",
            align_mode: str = "bilinear",
            **kwargs,
    ):
        super().__init__()
        ssmdims = dim if ssmdims is None else ssmdims

        self.normx = norm_layer(dim)
        self.normy = norm_layer(ssmdims)

        self.proj_y = nn.Identity() if ssmdims == dim else nn.Conv2d(ssmdims, dim, kernel_size=1, bias=True)
        self.fuse = GatedFusion(dim, gate_from=gate_from)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, kernel_size=1, bias=True),
            act_layer(),
            nn.Dropout(drop),
            nn.Conv2d(mlp_hidden_dim, dim, kernel_size=1, bias=True),
            nn.Dropout(drop),
        )
        self.norm2 = norm_layer(dim)
        self.align_mode = align_mode

    def _align_to(self, src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        return F.interpolate(src, size=ref.shape[-2:], mode=self.align_mode,
                             align_corners=False if self.align_mode in ("bilinear", "bicubic") else None)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        was_3d_x, was_3d_y = False, False

        if x.dim() == 3:
            was_3d_x = True
            B, L, C = x.shape
            H = int(math.sqrt(L))
            x = x.transpose(1, 2).reshape(B, C, H, H).contiguous()
        else:
            x = x.contiguous()

        if y.dim() == 3:
            was_3d_y = True
            B, L, C = y.shape
            H = int(math.sqrt(L))
            y = y.transpose(1, 2).reshape(B, C, H, H).contiguous()
        else:
            y = y.contiguous()

        nx = self.normx(x)
        ny = self.normy(y)
        ny = self.proj_y(ny)

        ny = self._align_to(ny, nx)

        fused = self.fuse(nx, ny)
        x = x + self.drop_path(fused)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if was_3d_x:
            B, C, H, W = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()

        return x