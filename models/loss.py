import paddle
import paddle.nn.functional as F


def _iou_loss(pred, target, smooth=1):
    intersection = paddle.sum(target * pred, axis=[1,2,3])
    union = paddle.sum(target, axis=[1,2,3]) + paddle.sum(pred, axis=[1,2,3])
    iou = paddle.mean((intersection + smooth) / (union + smooth), axis=0)
    return 1 - iou

def bce_ms_loss(pred, target, weight=[1, 0.8, 0.5, 0.5]):
    out, x8_out, x4_out, x2_out = tuple(pred)
    target = target.unsqueeze(1)
    target_2x = F.interpolate(
        target, x2_out.shape[2:], mode='bilinear', align_corners=True)
    target_4x = F.interpolate(
        target, x4_out.shape[2:], mode='bilinear', align_corners=True)
    target_8x = F.interpolate(
        target, x8_out.shape[2:], mode='bilinear', align_corners=True)
    
    _loss = F.binary_cross_entropy(F.sigmoid(out), target)
    _2x_loss = F.binary_cross_entropy(F.sigmoid(x2_out), target_2x)
    _4x_loss = F.binary_cross_entropy(F.sigmoid(x4_out), target_4x)
    _8x_loss = F.binary_cross_entropy(F.sigmoid(x8_out), target_8x)

    loss = weight[0] * _loss + weight[1] * _8x_loss + weight[2] * _4x_loss + \
         weight[3] * _2x_loss
    return loss

def bce_iou_ms_loss(pred, target, weight=[1, 0.8, 0.5, 0.5], weight2=1):
    out, x8_out, x4_out, x2_out = tuple(pred)
    target = target.unsqueeze(1)
    target_2x = F.interpolate(
        target, x2_out.shape[2:], mode='bilinear', align_corners=True)
    target_4x = F.interpolate(
        target, x4_out.shape[2:], mode='bilinear', align_corners=True)
    target_8x = F.interpolate(
        target, x8_out.shape[2:], mode='bilinear', align_corners=True)
    
    _loss = F.binary_cross_entropy(F.sigmoid(out), target) + \
        weight2 * _iou_loss(F.sigmoid(out), target)
    _2x_loss = F.binary_cross_entropy(F.sigmoid(x2_out), target_2x) + \
        weight2 * _iou_loss(F.sigmoid(x2_out), target_2x)
    _4x_loss = F.binary_cross_entropy(F.sigmoid(x4_out), target_4x) + \
        weight2 * _iou_loss(F.sigmoid(x4_out), target_4x)
    _8x_loss = F.binary_cross_entropy(F.sigmoid(x8_out), target_8x) + \
        weight2 * _iou_loss(F.sigmoid(x8_out), target_8x)

    loss = weight[0] * _loss + weight[1] * _8x_loss + weight[2] * _4x_loss + \
         weight[3] * _2x_loss
    return loss

def bce_iou_edge_ms_loss(pred, target, etarget, weight=[1, 0.8, 0.5, 0.5], weight2=1, weight3=1):
    out, x8_out, x4_out, x2_out,\
        edge, x8_edge, x4_edge, x2_edge = tuple(pred)
    
    # target
    target = target.unsqueeze(1)
    target_2x = F.interpolate(
        target, x2_out.shape[2:], mode='bilinear', align_corners=True)
    target_4x = F.interpolate(
        target, x4_out.shape[2:], mode='bilinear', align_corners=True)
    target_8x = F.interpolate(
        target, x8_out.shape[2:], mode='bilinear', align_corners=True)

    # edge target
    etarget = etarget.unsqueeze(1)
    etarget_2x = F.interpolate(
        etarget, x2_edge.shape[2:], mode='bilinear', align_corners=True)
    etarget_4x = F.interpolate(
        etarget, x4_edge.shape[2:], mode='bilinear', align_corners=True)
    etarget_8x = F.interpolate(
        etarget, x8_edge.shape[2:], mode='bilinear', align_corners=True)
    
    # loss
    _loss = F.binary_cross_entropy(F.sigmoid(out), target) + \
        weight2 * _iou_loss(F.sigmoid(out), target) + \
        weight3 * F.binary_cross_entropy(F.sigmoid(edge), etarget)
    _2x_loss = F.binary_cross_entropy(F.sigmoid(x2_out), target_2x) + \
        weight2 * _iou_loss(F.sigmoid(x2_out), target_2x) + \
        weight3 * F.binary_cross_entropy(F.sigmoid(x2_edge), etarget_2x)
    _4x_loss = F.binary_cross_entropy(F.sigmoid(x4_out), target_4x) + \
        weight2 * _iou_loss(F.sigmoid(x4_out), target_4x) + \
        weight3 * F.binary_cross_entropy(F.sigmoid(x4_edge), etarget_4x)
    _8x_loss = F.binary_cross_entropy(F.sigmoid(x8_out), target_8x) + \
        weight2 * _iou_loss(F.sigmoid(x8_out), target_8x) + \
        weight3 * F.binary_cross_entropy(F.sigmoid(x8_edge), etarget_8x)

    loss = weight[0] * _loss + weight[1] * _8x_loss + weight[2] * _4x_loss + \
         weight[3] * _2x_loss
    return loss
