import paddle
from paddlenlp.embeddings import TokenEmbedding
import paddle.nn as nn

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
