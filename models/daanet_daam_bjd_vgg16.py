import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from models.vgg import VGG16


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
        self.conv = DoubleConv2D(in_channels, out_channels)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out

class DoubleConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._res = in_channels == out_channels
        self.conv1 = BasicConv2D(in_channels, out_channels, 3, 1)
        self.conv2 = nn.Conv2D(out_channels, out_channels, 3, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn2(self.conv2(out))
        if self._res:
            out = out + x
        out = F.relu(out)
        return out

class ChannelAttention(nn.Layer):
    """
    Channel Attention Module
    """
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(in_channels, in_channels//ratio, 1, bias_attr=False)
        self.fc2 = nn.Conv2D(in_channels//ratio, in_channels, 1, bias_attr=False)
        
    def forward(self, x):
        pool_out = self.avgpool(x)
        avg_out = self.fc2(F.relu(self.fc1(pool_out)))
        avg_out2 = self.fc2(F.relu(self.fc1(pool_out)))
        out = avg_out + avg_out2
        return F.sigmoid(out)

class SpatialAttention(nn.Layer):
    """
    Spatial Attention Module
    """
    def __init__(self, kernel_size=7):
        super().__init__()

        assert kernel_size in (3, 7), 'kernel size should be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2D(2, 1, kernel_size, padding=padding, bias_attr=False)
        
    def forward(self, x):
        avg_out = paddle.mean(x, axis=1, keepdim=True)
        max_out = paddle.max(x, keepdim=True, axis=1)
        out = paddle.concat([avg_out, max_out], axis=1)
        out = self.conv(out)
        return F.sigmoid(out)

class BJD(nn.Layer):
    """
    Boundary-joint decoder
    """
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()

        self.reduce1 = BasicConv2D(in_channels, in_channels//2, 1)
        self.reduce2 = BasicConv2D(in_channels, in_channels//2, 1)

        self.up = Up(in_channels, out_channels)
        self.res = DoubleConv2D(out_channels, out_channels)
        self.classifier = nn.Conv2D(out_channels, 1, 3, 1, 1)

    def forward(self, x, shortcut):
        x_rd = self.reduce1(x)
        shortcut_rd = self.reduce2(shortcut)
        cat_out = paddle.concat([x_rd, shortcut_rd], axis=1)
        out = self.res(self.up(cat_out))
        return out, self.classifier(out)

class DAAM(nn.Layer):
    """
    Dual Attention Aggregation Module
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduce1 = BasicConv2D(in_channels, in_channels//2, 1)
        self.reduce2 = BasicConv2D(in_channels, in_channels//4, 1)
        self.increase1 = BasicConv2D(1, in_channels//4, 1)

        self.res = DoubleConv2D(in_channels, in_channels)
        self.cam = ChannelAttention(in_channels)
        self.sam = SpatialAttention()

        self.upsample = Up(in_channels, out_channels)

        self.dp = nn.Dropout2D(p=0.2)
        self.classifier = nn.Conv2D(out_channels, 1, 3, 1, 1)

    def forward(self, x, shortcut, seg, edge):
        addition = x + shortcut
        mult = addition * seg
        addition_rd = self.reduce1(addition)
        mult_rd = self.reduce2(mult)
        edge_ic = self.increase1(edge)
        out = self.res(paddle.concat([addition_rd, mult_rd, edge_ic], axis=1))
        cam_att = self.cam(out)
        out = cam_att * out
        sam_att = self.sam(out)
        out = sam_att * out
        out = self.upsample(out)
        seg = self.classifier(self.dp(out))
        return out, seg

class DAANet(nn.Layer):
    def __init__(self, pretrained=True):
        super().__init__()

        # encoder/backbone
        self.vgg16 = VGG16(pretrained=pretrained)

        # edge decoder
        self.edec1 = nn.Sequential(
            DoubleConv2D(512, 512), Up(512, 256))
        self.edec1_classifier = nn.Sequential(
            nn.Conv2D(256, 1, 3, 1, 1), nn.Dropout2D(p=0.2))
        self.edec1_rd = BasicConv2D(512, 256, 1)
        self.edec2 = BJD(256, 128)
        self.edec3 = BJD(128, 64)
        self.edec4 = BJD(64, 64)

        # semantic decoder
        self.dec1 = nn.Sequential(
            DoubleConv2D(512, 512), Up(512, 256))
        self.dec1_classifier = nn.Sequential(
            nn.Conv2D(256, 1, 3, 1, 1), nn.Dropout2D(p=0.2))
        self.dec2 = DAAM(256, 128)
        self.dec3 = DAAM(128, 64)
        self.dec4 = DAAM(64, 64)

    def forward(self, x):
        # encode
        feat1, feat2, feat3, feat4, output = self.vgg16(x)
        
        # edge decode
        edge = self.edec1(paddle.concat([
            self.edec1_rd(output), self.edec1_rd(feat4)], axis=1))
        x2_edge = self.edec1_classifier(edge)
        edge, x4_edge = self.edec2(edge, feat3)
        edge, x8_edge = self.edec3(edge, feat2)
        _, final_edge = self.edec4(edge, feat1)

        # semantic decode
        out = self.dec1(output+feat4)
        x2_out = self.dec1_classifier(out)
        out, x4_out = self.dec2(out, feat3, x2_out, F.sigmoid(x2_edge))
        out, x8_out = self.dec3(out, feat2, x4_out, F.sigmoid(x4_edge))
        _, final_out = self.dec4(out, feat1, x8_out, F.sigmoid(x8_edge))

        return final_out, x8_out, x4_out, x2_out, \
            final_edge, x8_edge, x4_edge, x2_edge