import paddle

def mae(pred, label):
    return float(paddle.mean(paddle.abs(label-pred)))
