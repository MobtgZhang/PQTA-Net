import paddle
import paddle.nn as nn

from paddlenlp.transformers import ErnieModel

class ErnieMReader(nn.Layer):
    def __init__(self):
        super(ErnieMReader, self).__init__()
    def forward(self, *inputs, **kwargs):
        pass
