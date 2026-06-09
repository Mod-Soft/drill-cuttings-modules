import math
import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward


class FAW_ECA(nn.Module):

    def __init__(self, channels, kernel_size=None, gamma=2.0, b=1.0):
        super().__init__()

        if kernel_size is None:
            t = int(abs((math.log2(channels) / gamma) + b))
            k = t if (t % 2 == 1) else (t + 1)
        else:
            k = kernel_size

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)

        y = y.squeeze(-1).transpose(-1, -2)

        y = self.conv(y)
        y = self.sigmoid(y)

        y = y.transpose(-1, -2).unsqueeze(-1)

        return x * y.expand_as(x)


class FrequencyAttentiveWavelet(nn.Module):

    def __init__(self, in_ch, out_ch, eca_kernel_size=None, eca_gamma=2.0, eca_b=1.0):
        super().__init__()
        if out_ch == in_ch:
            out_ch = in_ch * 2

        self.wt = DWTForward(J=1, mode='zero', wave='haar')

        concat_ch = in_ch * 4

        self.att = FAW_ECA(
            channels=concat_ch,
            kernel_size=eca_kernel_size,
            gamma=eca_gamma,
            b=eca_b
        )

        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(concat_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        orig_dtype = x.dtype
        was_3d = False
        is_swin_bhwc = False

        if x.dim() == 3:
            was_3d = True
            B, L, C = x.shape
            H = int(math.sqrt(L))
            x = x.transpose(1, 2).reshape(B, C, H, H).contiguous()
        elif x.dim() == 4 and x.shape[1] == x.shape[2] and x.shape[3] != x.shape[1]:
            is_swin_bhwc = True
            x = x.permute(0, 3, 1, 2).contiguous()
        else:
            x = x.contiguous()

        if torch.is_autocast_enabled():
            if x.device.type == "cuda":
                autocast_ctx = torch.cuda.amp.autocast(enabled=False)
            else:
                autocast_ctx = torch.autocast(device_type=x.device.type, enabled=False)
            with autocast_ctx:
                yL, yH = self.wt(x.float())
        else:
            yL, yH = self.wt(x)

        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]

        x_cat = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)

        if x_cat.dtype != orig_dtype:
            x_cat = x_cat.to(dtype=orig_dtype)

        x_weighted = self.att(x_cat)

        out = self.conv_bn_relu(x_weighted)

        if was_3d:
            B, C, H, W = out.shape
            out = out.flatten(2).transpose(1, 2).contiguous()
        elif is_swin_bhwc:
            out = out.permute(0, 2, 3, 1).contiguous()
        else:
            out = out.contiguous()

        return out