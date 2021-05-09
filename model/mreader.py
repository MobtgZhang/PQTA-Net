import paddle
import paddle.nn as nn

from .embedding import Embedding,Encoder
from .layers import SeqAttnMatch,SelfAttnMatch
from .layers import SFU,StackedBRNN
from .layers import MemoryAnsPointer

class MReader(nn.Layer):
    def __init__(self,args):
        super(MReader, self).__init__()

        self.embedding_name = args.embedding_name

        self.hop_layers = args.hop_layers
        self.embedding = Embedding(args.embedding_name)
        self.embedding_dim = self.embedding.embedding_dim
        self.hidden_size = args.hidden_size
        self.encoder = Encoder(self.embedding_dim,self.hidden_size)
        self.embedding_dim = self.embedding.embedding_dim
        question_hidden_size = self.hidden_size
        self.qt_attn = SeqAttnMatch(question_hidden_size)
        self.interactive_aligners = nn.LayerList()
        self.interactive_SFUs = nn.LayerList()
        self.self_aligners = nn.LayerList()
        self.self_SFUs = nn.LayerList()
        self.aggregate_rnns = nn.LayerList()
        doc_hidden_size = self.hidden_size
        for i in range(self.hop_layers):
            self.interactive_aligners.append(SeqAttnMatch(doc_hidden_size,identity=True))
            self.interactive_SFUs.append(SFU(doc_hidden_size,3*doc_hidden_size))
            # self aligner
            self.self_aligners.append(SelfAttnMatch(doc_hidden_size,identity=True,diag=False))
            self.self_SFUs.append(SFU(doc_hidden_size,doc_hidden_size*3))
            # aggregating
            self.aggregate_rnns.append(
                StackedBRNN(
                    input_size=doc_hidden_size,
                    hidden_size=doc_hidden_size//2,
                    num_layers=1,
                    dropout_rate=args.dropout_rate,
                    dropout_output=args.dropout_rnn,
                    concat_layers=False,
                    rnn_type=nn.GRU
                )
            )
        self.ptr_ans_net = MemoryAnsPointer(x_size=doc_hidden_size,
                                            y_size=question_hidden_size,
                                            hidden_size=self.hidden_size,
                                            hop=self.hop_layers,
                                            dropout_rate=args.dropout_rate,
                                            normalize=args.normalize)
    def forward(self,docs_c,docs_w,tite_c,tite_w,ques_c,ques_w):
        # embedding the layers
        tite_emb = self.embedding(tite_c, tite_w)
        ques_emb = self.embedding(ques_c, ques_w)
        docs_emb = self.embedding(docs_c, docs_w)
        tite_emb = self.encoder(tite_emb)
        ques_emb = self.encoder(ques_emb)
        docs_emb = self.encoder(docs_emb)
        # Align and aggregate
        qt_hid_attn = self.qt_attn(ques_emb, tite_emb)
        c_check = docs_emb
        for i in range(self.hop_layers):
            if i%2 == 0:
                qt_tilde = self.interactive_aligners[i].forward(c_check, ques_emb)
            else:
                qt_tilde = self.interactive_aligners[i].forward(c_check,qt_hid_attn)
            c_bar = self.interactive_SFUs[i].forward(c_check,paddle.concat([qt_tilde,c_check*qt_tilde,c_check-qt_tilde],axis=2))

            c_tilde = self.self_aligners[i].forward(c_bar)
            c_hat = self.self_SFUs[i].forward(c_bar,paddle.concat([c_tilde,c_bar*c_tilde,c_bar-c_tilde],axis=2))
            c_check = self.aggregate_rnns[i].forward(c_hat)
        # predict Pointer Network
        start_pos,end_pos,cls_pos = self.ptr_ans_net(c_check,ques_emb)
        return start_pos,end_pos,cls_pos


















