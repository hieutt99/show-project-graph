import torch
import torch.nn as nn


def conv(cin, cout, kernel):
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel, stride=1, padding=kernel//2, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(True)
    )


class DoubleConv(nn.Module):
    def __init__(self, cin, cout, res=True):
        super().__init__()

        self.res = cin == cout and res

        self.layers = nn.Sequential(
            conv(cin, cout // 2, 1),
            conv(cout // 2, cout, 3)
        )

    def forward(self, x):
        if self.res:
            return x + self.layers(x)
        else:
            return self.layers(x)


class DoubleConvChain(nn.Module):
    def __init__(self, n, cin, cout, res=True):
        super().__init__()

        layers = []
        for _ in range(n):
            layers.append(DoubleConv(cin, cout, res))
            cin = cout

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def down_sample(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 2, stride=2, padding=0),
        nn.ReLU(True)
    )


def up_sample(cin, cout):
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, kernel_size=2, stride=2),
        nn.ReLU(True)
    )


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.inc = nn.Sequential(
            conv(3, 32, 3),
            conv(32, 32, 3)
        )
        self.down1 = down_sample(32, 64)
        self.layers1 = DoubleConv(64, 64)

        self.down2 = down_sample(64, 128)
        self.layers2 = DoubleConvChain(2, 128, 128)

        self.down3 = down_sample(128, 256)
        self.layers3 = DoubleConvChain(2, 256, 256)

        self.down4 = down_sample(256, 512)
        self.layers4 = DoubleConvChain(2, 512, 512)

        #####################################

        self.up1 = up_sample(512, 256)
        self.layers5 = DoubleConvChain(2, 512, 256, False)

        self.up2 = up_sample(256, 128)
        self.layers6 = DoubleConvChain(2, 256, 128, False)

        self.up3 = up_sample(128, 64)
        self.layers7 = DoubleConvChain(2, 128, 64, False)

        self.up4 = up_sample(64, 64)
        self.layers8 = DoubleConvChain(2, 96, 64, False)

        self.s_head = nn.Conv2d(64, 1, 1)

        self.h_head = nn.Sequential(
            DoubleConvChain(2, 128, 128, res=False),
            nn.Conv2d(128, 1, 1),
            nn.Upsample(scale_factor=4, mode='nearest')
        )

    def forward(self, x):
        cache = []

        x = self.inc(x)
        cache.append(x)

        x = self.down1(x)
        x = self.layers1(x)
        cache.append(x)

        x = self.down2(x)
        x = self.layers2(x)
        cache.append(x)

        x = self.down3(x)
        x = self.layers3(x)
        cache.append(x)

        x = self.down4(x)
        x = self.layers4(x)

        ##########################

        x = self.up1(x)
        x = torch.cat((x, cache.pop(-1)), dim=1)
        x = self.layers5(x)

        x = self.up2(x)
        x = torch.cat((x, cache.pop(-1)), dim=1)
        x = self.layers6(x)
        h = x

        x = self.up3(x)
        x = torch.cat((x, cache.pop(-1)), dim=1)
        x = self.layers7(x)

        x = self.up4(x)
        x = torch.cat((x, cache.pop(-1)), dim=1)
        x = self.layers8(x)

        s = self.s_head(x)
        h = self.h_head(h)

        return torch.cat([s, h], dim=1)
