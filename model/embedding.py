import paddle
import paddle.nn.functional as F
from paddlenlp.embeddings import TokenEmbedding
import paddle.nn as nn

class Encoder(nn.Layer):
    def __init__(self,input_size,hidden_size):
        super(Encoder, self).__init__()
        weight_attr_r = paddle.framework.ParamAttr(initializer=nn.initializer.XavierNormal())
        bias_attr_r = paddle.framework.ParamAttr(initializer=nn.initializer.XavierNormal())
        weight_attr_g = paddle.framework.ParamAttr(initializer=nn.initializer.XavierNormal())
        bias_attr_g = paddle.framework.ParamAttr(initializer=nn.initializer.XavierNormal())
        self.linear_r = nn.Linear(input_size, hidden_size, weight_attr=weight_attr_r,
                                  bias_attr=bias_attr_r)
        self.linear_g = nn.Linear(input_size, hidden_size, weight_attr=weight_attr_g,
                                  bias_attr=bias_attr_g)
        weight_attr = paddle.framework.ParamAttr(initializer=nn.initializer.XavierNormal())
        self.weights = nn.Linear(input_size,hidden_size,weight_attr=weight_attr)
    def forward(self,inputs_emb):
        r = self.linear_r(inputs_emb)
        g = self.linear_g(inputs_emb)
        r,g = F.tanh(r),F.sigmoid(g)
        o = g*r + (1-g)*self.weights(inputs_emb)
        return o
class Embedding(nn.Layer):
    def __init__(self,embedding_name):
        super(Embedding, self).__init__()
        self.embedding =TokenEmbedding(embedding_name)
        self.embedding_dim = self.embedding.embedding_dim
        weight_attr = paddle.framework.ParamAttr(
            name="linear_weight",
            initializer=paddle.nn.initializer.XavierNormal())
        bias_attr = paddle.framework.ParamAttr(
            name="linear_bias",
            initializer=paddle.nn.initializer.XavierNormal())
        self.mlp = paddle.nn.Linear(self.embedding_dim*2, self.embedding_dim, weight_attr=weight_attr, bias_attr=bias_attr)
        self.gru = nn.GRU(input_size=self.embedding_dim,hidden_size=self.embedding_dim//2,num_layers=1,
                          direction="bidirectional",)
    def forward(self, chars_ids,words_ids):
        chars_emb = self.embedding(chars_ids)
        words_emb = self.embedding(words_ids)
        extrs_emb = paddle.concat([chars_emb,words_emb],axis=2)
        extrs_emb = self.mlp(extrs_emb)
        extrs_emb,_ = self.gru(extrs_emb)
        return extrs_emb
