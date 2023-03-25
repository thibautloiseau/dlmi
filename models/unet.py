import torch
from torch import nn
import torch.nn.functional as F


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetDown, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.GroupNorm(16, out_size),
            nn.SiLU(),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            nn.GroupNorm(16, out_size),
            nn.SiLU(),
            nn.MaxPool2d(2)
          )

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(UNetUp, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.GroupNorm(16, out_size),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1),
            nn.GroupNorm(16, out_size),
            nn.SiLU()
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x


class FinalLayer(nn.Module):
    def __init__(self, in_size, out_size):
        super(FinalLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_size)
        )

    def forward(self, x, skip_input=None):
        if skip_input is not None:
            x = torch.cat((x, skip_input), 1)  # add the skip connection
        x = self.model(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=11, out_channels=1):
        super(UNet, self).__init__()

        self.down1 = UNetDown(in_channels, 32)
        self.down2 = UNetDown(32, 64)
        self.down3 = UNetDown(64, 128)
        self.down4 = UNetDown(128, 256)
        self.down5 = UNetDown(256, 256)

        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)
        self.up4 = UNetUp(128, 32)

        self.final = FinalLayer(64, out_channels)

    def forward(self, x, possible_dose_mask):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u2 = self.up2(u1, d4)
        u3 = self.up3(u2, d3)
        u4 = self.up4(u3, d2)

        output = 100*self.final(u4, d1)*possible_dose_mask

        return output
