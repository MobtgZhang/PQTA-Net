import paddle
import paddle.nn as nn

from paddlenlp.transformers import ErnieModel

class ErnieRNet(nn.Layer):
    def __init__(self):
        super(ErnieRNet, self).__init__()
    def forward(self, *inputs, **kwargs):
        pass