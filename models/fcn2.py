import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Upsample(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.upsample(x)


class Fuse(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Fuse, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fuse(x)


class Resnet2(nn.Module):
    def __init__(self, num_classes):
        super(Resnet2, self).__init__()

        self.num_classes = num_classes

        resnet = models.resnet101(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.upsample4 = Upsample(2048, 2048, 1024)
        self.upsample3 = Upsample(2048, 1024, 512)
        self.upsample2 = Upsample(1024, 512, 256)
        self.fuse_fm1 = Fuse(512, 256, 64)
        self.upsample1 = Upsample(128, 64, 32)

        self.out = self._classifier(32)

    def _classifier(self, inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, self.num_classes, 1),
            nn.Conv2d(self.num_classes, self.num_classes,
                      kernel_size=3, padding=1)
        )

    def forward(self, x):
        x_copy = x.clone()
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x  # 64 112 112
        x = self.maxpool(x)
        pool_x = x  # 64 56 56
        # print (pool_x.size())
        fm1 = self.layer1(x)  # 256 56 56
        # print (fm1.size())
        fm2 = self.layer2(fm1)  # 512 28 28
        # print (fm2.size())
        fm3 = self.layer3(fm2)  # 1024 14 14
        # print (fm3.size())
        fm4 = self.layer4(fm3)  # 2048 7 7
        # print (fm4.size())

        # print ("*********")

        fs_fm4 = self.upsample4(fm4)  # 2048, 14, 14
        # print (fs_fm4.size())
        fs_fm3 = self.upsample3(torch.cat([fs_fm4, fm3], axis=1))
        # print (fs_fm3.size())
        fs_fm2 = self.upsample2(torch.cat([fs_fm3, fm2], axis=1))
        # print (fs_fm2.size())
        fs_fm1 = self.fuse_fm1(torch.cat([fs_fm2, fm1], axis=1))
        # print (fs_fm1.size())
        fs_pool = self.upsample1(torch.cat([fs_fm1, pool_x], axis=1))
        # print (fs_pool.size())
        temp = F.interpolate(fs_pool, x_copy.size()[2:], mode='bilinear', align_corners=True)
        # print (temp.size())
        out = self.out(temp)
        # print (out.size())

        out = F.log_softmax(out, dim=1)

        return out
