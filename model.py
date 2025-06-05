import torch.nn as nn
import torch
import torch.nn.functional as F

class MLSE(nn.Module):
    def __init__(self, model, feature_size):
        super(MLSE, self).__init__()

        self.features = model
        self.max3 = nn.MaxPool2d((25,25), stride=(25,25))
        self.avg1 = nn.AdaptiveAvgPool2d(1)
        self.avg2 = nn.AdaptiveAvgPool2d(1)
        self.num_ftrs = 2048 * 1 * 1

        self.classifier_concat1 = nn.Sequential(
            nn.BatchNorm1d(feature_size * 3),
            nn.Linear(feature_size* 3, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.classifier3 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ReLU(inplace=True),
        )

        # Smooth layers
        self.smooth1 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        # Lateral layers
        self.latlayer1 = nn.Conv2d(self.num_ftrs // 2, feature_size, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(self.num_ftrs // 4, feature_size, kernel_size=1, stride=1, padding=0)
        self.toplayer = nn.Conv2d(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0)
        self.pclassifier1 = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.ReLU(inplace=True),
        )
        self.pclassifier2 = nn.Sequential(
            nn.BatchNorm1d(feature_size),
            nn.ReLU(inplace=True),
        )

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        xf1, xf2, xf3, xf4, xf5 = self.features(x)

        xl3 = self.max3(xf5)
        xl3 = xl3.view(xl3.size(0), -1)
        xc3 = self.classifier3(xl3)
        # #
        # # #
        xl3 = self.toplayer(xf5)
        #
        p4 = self._upsample_add(xl3, self.latlayer1(xf4))
        p4 = self.smooth1(p4)
        p3 = self._upsample_add(p4, self.latlayer2(xf3))
        p3 = self.smooth2(p3)

        xp1 = self.avg1(p3)
        xp1 = xp1.view(xp1.size(0), -1)
        xp1 = self.pclassifier1(xp1)

        xp2 = self.avg2(p4)
        xp2 = xp2.view(xp2.size(0), -1)
        xp2 = self.pclassifier2(xp2)

        z_weighted_avg = torch.cat((xp1,xp2,xc3), -1)
        x_concat1 = self.classifier_concat1(z_weighted_avg)
        return x_concat1


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
