###-------------------------------------
###               UNet
###-------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=False):
        super(UNetConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.pooling = pooling

    def forward(self, x):
        if self.pooling:
            x = F.max_pool2d(x, 2)
        out = self.conv(x)
        return out

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mode='upsample'):
        super(UNetUpBlock, self).__init__()

        if mode == 'upsample':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        out = self.conv(x)
        return out
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, n_classes=2, mode='upsample'):
        super(UNet, self).__init__()
        self.dconv1 = UNetConvBlock(in_channels, 64)
        self.dconv2 = UNetConvBlock(64, 128, pooling=True)
        self.dconv3 = UNetConvBlock(128, 256, pooling=True)
        self.dconv4 = UNetConvBlock(256, 512, pooling=True)
        factor = 2 if mode=='upsample' else 1
        self.dconv5 = UNetConvBlock(512, 1024//factor, pooling=True)
        self.uconv1 = UNetUpBlock(1024, 512//factor, mode=mode)
        self.uconv2 = UNetUpBlock(512, 256//factor, mode=mode)
        self.uconv3 = UNetUpBlock(256, 128//factor, mode=mode)
        self.uconv4 = UNetUpBlock(128, 64, mode=mode)
        self.out = nn.Conv2d(64, n_classes, 1)

        self.__init_parameters()

    def __init_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.dconv1(x)
        x2 = self.dconv2(x1)
        x3 = self.dconv3(x2)
        x4 = self.dconv4(x3)
        x5 = self.dconv5(x4)

        x = self.uconv1(x5, x4)
        x = self.uconv2(x, x3)
        x = self.uconv3(x, x2)
        x = self.uconv4(x, x1)
        out = self.out(x)
        return out

