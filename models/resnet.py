import paddle.nn as nn
from paddle.vision.models import resnet50


class ResNet50(nn.Layer):
    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone = resnet50(pretrained)

        # 5 x downsample --> 4 x downsample
        for n, l in self.backbone.layer4.named_sublayers():
            if n == '0.conv2' and isinstance(l, nn.Conv2D):
                l._stride = [1, 1]
                l._dilation = [2, 2]
                l._padding = [2, 2]
                l._updated_padding = [2, 2]
            if n == '0.downsample.0' and isinstance(l, nn.Conv2D):
                l._stride = [1, 1]

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)  # (N, 64, H/2, W/2)
        res0 = self.backbone.maxpool(x)

        res1 = self.backbone.layer1(res0)  # (N, 256, H/4, W/4)
        res2 = self.backbone.layer2(res1)  # (N, 512, H/8, W/8)
        res3 = self.backbone.layer3(res2)  # (N, 1024, H/16, W/16)
        res4 = self.backbone.layer4(res3)  # (N, 2048, H/16, W/16)

        return x, res1, res2, res3, res4