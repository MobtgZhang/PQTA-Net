import paddle.framework
import paddle.nn as nn
import paddle.nn.functional as F

from paddlenlp.embeddings import TokenEmbedding

from .layers import SeqAttnMatch,SelfAttnMatch,SFU
from .layers import AggrateBiRNN
class Encoder(nn.Layer):
    def __init__(self,input_size,hidden_size,dropout=0.2):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.recurrent_layer = nn.GRU(input_size=input_size,
                                      hidden_size=input_size,num_layers=1,direction="bidirectional")
        weight_attr1 = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierNormal())
        weight_attr2 = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierNormal())
        self.linear1 = nn.Linear(input_size*2,3*hidden_size,weight_attr=weight_attr1)
        self.linear2 = nn.Linear(input_size*2,3*hidden_size,weight_attr=weight_attr2)

    def forward(self,emb_words,emb_chars):
        words_encode,_ = self.recurrent_layer(emb_words)
        chars_encode,_ = self.recurrent_layer(emb_chars)
        seq_encode = F.relu(self.linear1(words_encode)+self.linear2(chars_encode))
        seq_encode = F.dropout(seq_encode,p=self.dropout,training=self.training)
        return seq_encode
class PQTANet(nn.Layer):
    def __init__(self,args):
        super(PQTANet, self).__init__()
        # Embedding Layer
        self.embedding = TokenEmbedding(args.embedding_name)
        self.embedding_dim = self.embedding.embedding_dim
        self.hidden_size = args.hidden_size
        # Encoder Layer
        self.emb_encoder = Encoder(self.embedding_dim,self.hidden_size)
        # Question-Answer Contact Layer
        hidden_out_size = self.hidden_size*3

        self.interactive_aligners = nn.LayerList()
        self.interactive_SFUs = nn.LayerList()
        self.interactive_qes_aligners = nn.LayerList()
        self.interactive_qes_SFUs = nn.LayerList()
        self.self_aligners = nn.LayerList()
        self.self_SFUs = nn.LayerList()
        self.aggregate_rnns = nn.LayerList()




        for i in range(self.hop_layers):
            self.interactive_aligners.append(SeqAttnMatch(hidden_out_size,identity=True))
            self.interactive_SFUs.append(SFU(hidden_out_size,3*hidden_out_size))
            self.interactive_qes_aligners.append(SeqAttnMatch(hidden_out_size,identity=True))
            self.interactive_qes_SFUs.append(SFU(hidden_out_size,3*hidden_out_size))
            # self aligner
            self.self_aligners.append(SelfAttnMatch(hidden_out_size,identity=True,diag=False))
            self.self_SFUs.append(SFU(hidden_out_size,hidden_out_size*3))
            # aggregating
            self.aggregate_rnns.append(
                AggrateBiRNN(

                )
            )
        '''
                            input_size=hidden_out_size,
                            hidden_size=hidden_out_size//2,
                            num_layers=1,
                            dropout_rate=args.dropout_rate,
                            dropout_output=args.dropout_rnn,
                            concat_layers=False,
                            rnn_type=nn.GRU
                            '''











        # Self-Attention Layer
        # Pointer Layer
    def forward(self,docs_c,docs_w,tite_c,tite_w,ques_c,ques_w):
        # Embedding Layer & Encoder Layer
        docs_seq_encode = self.emb_encoder(self.embedding(docs_w),self.embedding(docs_c))
        ques_seq_encode = self.emb_encoder(self.embedding(ques_w),self.embedding(ques_c))
        tite_seq_encode = self.emb_encoder(self.embedding(tite_w),self.embedding(tite_c))
        # contact layers
        doc_tit = self.docs_title_attn(docs_seq_encode,tite_seq_encode)

        print(docs_seq_encode.shape,ques_seq_encode.shape,tite_seq_encode.shape)
        exit()
