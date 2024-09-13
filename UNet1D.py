import torch
import torch.nn as nn
import torch.nn.functional as F


class ContractingBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, down_type='Pooling'):
        super(ContractingBlock1D, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.ReLU()
        )

        self.down_type = down_type
        if down_type == 'Pooling':
            self.down_sample = nn.MaxPool1d(2, stride=2)
        elif down_type == 'Conv':
            self.down_sample = nn.Conv1d(out_channels, out_channels, 2, stride=2,
                                         groups=out_channels)  # Depthwise convolution

    def forward(self, x):
        x = self.layer(x)
        x_down = self.down_sample(x)
        return x, x_down


class ExpansiveBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ExpansiveBlock1D, self).__init__()
        self.up_sample = nn.ConvTranspose1d(in_channels, out_channels, 2, stride=2)
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x, skip):
        x = self.up_sample(x)
        diff = skip.size(-1) - x.size(-1)
        x = F.pad(x, (diff // 2, diff - diff // 2))
        x = torch.cat([skip, x], dim=1)
        x = self.layer(x)
        return x


class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, down_type='Pooling'):
        super(UNet1D, self).__init__()
        self.contract1 = ContractingBlock1D(in_channels, 64, down_type)
        self.contract2 = ContractingBlock1D(64, 128, down_type)
        self.contract3 = ContractingBlock1D(128, 256, down_type)
        self.contract4 = ContractingBlock1D(256, 512, down_type)
        self.contract5 = ContractingBlock1D(512, 1024, down_type)

        self.expand4 = ExpansiveBlock1D(1024, 512)
        self.expand3 = ExpansiveBlock1D(512, 256)
        self.expand2 = ExpansiveBlock1D(256, 128)
        self.expand1 = ExpansiveBlock1D(128, 64)

        self.final_conv = nn.Conv1d(64, out_channels, 1)

    def forward(self, x):
        x1, x1_down = self.contract1(x)
        x2, x2_down = self.contract2(x1_down)
        x3, x3_down = self.contract3(x2_down)
        x4, x4_down = self.contract4(x3_down)
        x5, x5_down = self.contract5(x4_down)

        x = self.expand4(x5, x4)
        x = self.expand3(x, x3)
        x = self.expand2(x, x2)
        x = self.expand1(x, x1)

        x = self.final_conv(x)
        return x
