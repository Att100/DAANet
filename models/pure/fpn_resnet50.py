import paddle.nn as nn
import paddle.nn.functional as F

from models.resnet import ResNet50


class BasicConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1):
        super().__init__()

        self.conv2d = nn.Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        out = self.conv2d(x)
        out = F.relu(out)
        return self.bn(out)

class Up(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = BasicConv2D(in_channels, out_channels, 3, 1)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out

class DoubleConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = BasicConv2D(in_channels, out_channels, 1)
        self.conv2 = BasicConv2D(out_channels, out_channels, 3, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out

class FPN(nn.Layer):
    def __init__(self, pretrained=True):
        super().__init__()

        # encoder/backbone
        self.resnet50 = ResNet50(pretrained=pretrained)
        self.reduce = BasicConv2D(2048, 1024, 1)

        # decoder
        self.up1 = Up(1024, 512)
        self.updc1 = DoubleConv2D(1024, 1024)
        self.up2 = Up(512, 256)
        self.updc2 = DoubleConv2D(512, 512)
        self.up3 = Up(256, 64)
        self.updc3 = DoubleConv2D(256, 256)
        self.up4 = Up(64, 64)
        self.updc4 = DoubleConv2D(64, 64)

        # classifier
        self.dp = nn.Dropout2D(p=0.5)
        self.classifier = nn.Conv2D(64, 1, 3, 1, 1)

        # deep supervision classifier
        self.dp_x8 = nn.Dropout2D(p=0.2)
        self.dp_x4 = nn.Dropout2D(p=0.2)
        self.dp_x2 = nn.Dropout2D(p=0.2)
        self.classifier_x8 = nn.Conv2D(64, 1, 3, 1, 1)
        self.classifier_x4 = nn.Conv2D(256, 1, 3, 1, 1)
        self.classifier_x2 = nn.Conv2D(512, 1, 3, 1, 1)


    def forward(self, x):
        # encode
        feat1, feat2, feat3, feat4, output = self.resnet50(x)

        #decode
        add1 = self.updc1(self.reduce(output)+feat4)
        out_x2 = self.up1(add1)
        add2 = self.updc2(out_x2+feat3)
        out_x4 = self.up2(add2)
        add3 = self.updc3(out_x4+feat2)
        out_x8 = self.up3(add3)
        add4 = self.updc4(out_x8+feat1)
        out = self.up4(add4)
        
        # auxiliary outputs
        x8_out = self.classifier_x8(self.dp_x8(out_x8))
        x4_out = self.classifier_x4(self.dp_x4(out_x4))
        x2_out = self.classifier_x2(self.dp_x2(out_x2))

        # final output
        out = self.classifier(self.dp(out))

        return out, x8_out, x4_out, x2_out

