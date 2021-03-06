import paddle
import paddle.nn as nn
class CrossEntropyLoss(nn.Layer):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
    def forward(self, predict,target):
        start_logits, end_logits,cls_logits = predict
        start_position, end_position, answerable_label = target
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        answerable_label = paddle.unsqueeze(answerable_label, axis=-1)
        start_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=start_logits, label=start_position, soft_label=False)
        start_loss = paddle.mean(start_loss)
        end_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=end_logits, label=end_position, soft_label=False)
        end_loss = paddle.mean(end_loss)
        cls_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=cls_logits, label=answerable_label, soft_label=False)
        cls_loss = paddle.mean(cls_loss)
        mrc_loss = (start_loss + end_loss) / 2
        loss = (mrc_loss + cls_loss) /2
        return loss
class CrossEntropyLossWithOutCls(nn.Layer):
    def __init__(self):
        super(CrossEntropyLossWithOutCls, self).__init__()
    def forward(self, predict,target):
        start_logits, end_logits,_ = predict
        start_position, end_position, _ = target
        start_position = paddle.unsqueeze(start_position, axis=-1)
        end_position = paddle.unsqueeze(end_position, axis=-1)
        start_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=start_logits, label=start_position, soft_label=False)
        start_loss = paddle.mean(start_loss)
        end_loss = paddle.nn.functional.softmax_with_cross_entropy(
            logits=end_logits, label=end_position, soft_label=False)
        end_loss = paddle.mean(end_loss)
        loss = (start_loss + end_loss) / 2
        return loss