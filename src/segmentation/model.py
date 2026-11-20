
import torch
import torch.nn as nn
import torchvision.models as models


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)


class UNetResNet18(nn.Module):

    def __init__(self, pretrained=False):

        super().__init__()

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = models.resnet18(weights=weights)

        self.encoder1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu
        )

        self.maxpool = backbone.maxpool
        self.encoder2 = backbone.layer1
        self.encoder3 = backbone.layer2
        self.encoder4 = backbone.layer3
        self.encoder5 = backbone.layer4

        self.center = DoubleConv(512, 256)

        self.decoder4 = DoubleConv(256 + 256, 128)
        self.decoder3 = DoubleConv(128 + 128, 64)
        self.decoder2 = DoubleConv(64 + 64, 64)
        self.decoder1 = DoubleConv(64 + 64, 32)

        self.final = nn.Conv2d(
            32,
            1,
            kernel_size=1
        )

    def forward(self, x):

        input_size = x.shape[2:]

        e1 = self.encoder1(x)
        e2 = self.encoder2(self.maxpool(e1))
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)

        center = self.center(e5)

        d4 = torch.nn.functional.interpolate(
            center,
            size=e4.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d4 = self.decoder4(
            torch.cat([d4, e4], dim=1)
        )

        d3 = torch.nn.functional.interpolate(
            d4,
            size=e3.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d3 = self.decoder3(
            torch.cat([d3, e3], dim=1)
        )

        d2 = torch.nn.functional.interpolate(
            d3,
            size=e2.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d2 = self.decoder2(
            torch.cat([d2, e2], dim=1)
        )

        d1 = torch.nn.functional.interpolate(
            d2,
            size=e1.shape[2:],
            mode="bilinear",
            align_corners=False
        )

        d1 = self.decoder1(
            torch.cat([d1, e1], dim=1)
        )

        output = self.final(d1)

        output = torch.nn.functional.interpolate(
            output,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )

        return output
