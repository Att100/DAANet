import paddle.nn as nn
import paddle.nn.functional as F

from models.mobilenet_v2 import MobileNetV2


class BasicConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()

        self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        out = self.conv2d(x)
        out = F.relu(out)
        return self.bn(out)

class DoubleConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = BasicConv2D(in_channels, out_channels, 1)
        self.conv2 = BasicConv2D(out_channels, out_channels, 3, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class Up(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv2D(in_channels, out_channels)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out

class FPN(nn.Layer):
    def __init__(self, pretrained=True):
        super().__init__()

        # encoder/backbone
        self.mobilenet_v2 = MobileNetV2(pretrained=pretrained)

        # conv 1x1
        self.conv4 = BasicConv2D(64, 1280, 1)
        self.conv3 = BasicConv2D(32, 64, 1)
        self.conv2 = BasicConv2D(24, 32, 1)
        self.conv1 = BasicConv2D(32, 24, 1)

        # decoder
        self.up1 = Up(1280, 64)
        self.updc1 = DoubleConv2D(1280, 1280)
        self.up2 = Up(64, 32)
        self.updc2 = DoubleConv2D(64, 64)
        self.up3 = Up(32, 24)
        self.updc3 = DoubleConv2D(32, 32)
        self.up4 = Up(24, 32)
        self.updc4 = DoubleConv2D(24, 24)

        # classifier
        self.dp = nn.Dropout2D(p=0.5)
        self.classifier = nn.Conv2D(32, 1, 3, 1, 1)

        # deep supervision classifier
        self.dp_x8 = nn.Dropout2D(p=0.2)
        self.dp_x4 = nn.Dropout2D(p=0.2)
        self.dp_x2 = nn.Dropout2D(p=0.2)
        self.classifier_x8 = nn.Conv2D(24, 1, 3, 1, 1)
        self.classifier_x4 = nn.Conv2D(32, 1, 3, 1, 1)
        self.classifier_x2 = nn.Conv2D(64, 1, 3, 1, 1)

    def forward(self, x):
        # encode
        feat1, feat2, feat3, feat4, output = self.mobilenet_v2(x)

        feat4 = self.conv4(feat4)
        feat3 = self.conv3(feat3)
        feat2 = self.conv2(feat2)
        feat1 = self.conv1(feat1)

        #decode
        add1 = self.updc1(output + feat4)
        out_x2 = self.up1(add1)
        add2 = self.updc2(out_x2 + feat3)
        out_x4 = self.up2(add2)
        add3 = self.updc3(out_x4 + feat2)
        out_x8 = self.up3(add3)
        add4 = self.updc4(out_x8 + feat1)
        out = self.up4(add4)
        
        # auxiliary outputs
        x8_out = self.classifier_x8(self.dp_x8(out_x8))
        x4_out = self.classifier_x4(self.dp_x4(out_x4))
        x2_out = self.classifier_x2(self.dp_x2(out_x2))

        # final output
        out = self.classifier(self.dp(out))

        return out, x8_out, x4_out, x2_out