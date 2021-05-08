import paddle
import paddle.nn as nn
from paddlenlp.embeddings import TokenEmbedding
class Embedding(TokenEmbedding):
    def __init__(self):
        super(Embedding, self).__init__()
        self.gru = nn.GRU(self.embedding_dim,self.embedding_dim//2,direction="bidirect",num_layers=1)
    def forward(self,doc_w,doc_c,que_w,que_c,title_w,title_c,):
        pass
