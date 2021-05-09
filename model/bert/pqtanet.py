import paddle
import paddle.nn as nn

from paddlenlp.transformers import ErnieModel

class ErniePQTANet(nn.Layer):
    def __init__(self):
        super(ErniePQTANet, self).__init__()
    def forward(self, *inputs, **kwargs):
        pass
