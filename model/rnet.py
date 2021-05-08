import paddle.nn as nn

from .embedding import Embedding
from .layers import SeqAttnMatch,Gate,StackedBRNN,SelfAttnMatch,PointerNetwork

class RNet(nn.Layer):
    def __init__(self,args):
        super(RNet, self).__init__()
        self.embedding = Embedding(args.embedding_name)
        question_hidden_size = self.embedding.embedding_dim
        # question  title attention
        self.qt_attn = SeqAttnMatch(question_hidden_size)
        self.qt_gate = Gate(question_hidden_size)
        # qt document attention
        self.doc_attn = SeqAttnMatch(question_hidden_size)
        self.doc_attn_gate = Gate(question_hidden_size)
        doc_hidden_size = question_hidden_size//3
        self.doc_num_layers = 3
        self.doc_attn_rnn = StackedBRNN(input_size=question_hidden_size,
                                        hidden_size=doc_hidden_size,
                                        num_layers=self.doc_num_layers,
                                        dropout_rate=args.dropout_rate,
                                        rnn_type=nn.GRU)
        # document self attention
        doc_self_attn_size = (self.doc_num_layers-1)*doc_hidden_size
        self.doc_self_attn = SelfAttnMatch(input_size=doc_self_attn_size)
        self.doc_self_gate = Gate(input_size=doc_self_attn_size)
        doc_self_hidden_size = doc_self_attn_size//4
        self.doc_self_layers = 3
        self.doc_self_attn_rnn = StackedBRNN(input_size=doc_self_attn_size,
                                             hidden_size=doc_self_hidden_size,
                                             num_layers=self.doc_self_layers,
                                             dropout_rate=args.dropout_rate,
                                             rnn_type=nn.GRU)
        out_hidden_size = doc_self_hidden_size*(self.doc_self_layers-1)//2
        ptr_hidden_size = doc_self_hidden_size*(self.doc_self_layers-1)
        # Pointer Network
        self.ptr_net = PointerNetwork(
            x_size=ptr_hidden_size,
            y_size=question_hidden_size,
            hidden_size=out_hidden_size,
            dropout_rate=args.dropout_rnn,
            cell_type=nn.GRUCell,
            normalize=args.normalize
        )
    def forward(self,docs_c,docs_w,tite_c,tite_w,ques_c,ques_w):
        # embedding the layers
        tite_emb = self.embedding(tite_c,tite_w)
        ques_emb = self.embedding(ques_c,ques_w)
        docs_emb = self.embedding(docs_c,docs_w)
        # Context-Question Attention layer
        qt_hid_attn = self.qt_attn(ques_emb,tite_emb)
        qt_hid_attn = self.qt_gate(qt_hid_attn)
        cq_hid_attn = self.doc_attn(docs_emb,qt_hid_attn)
        cq_hid_attn = self.doc_attn_gate(cq_hid_attn)
        cq_hid_attn = self.doc_attn_rnn(cq_hid_attn)
        # docmentation self attention
        doc_hid_attn = self.doc_self_attn(cq_hid_attn)
        doc_hid_attn = self.doc_self_gate(doc_hid_attn)
        doc_hid_attn = self.doc_self_attn_rnn(doc_hid_attn)
        # Network pointer
        start_pointer,end_pointer,cls_logits = self.ptr_net(doc_hid_attn,ques_emb)
        return start_pointer,end_pointer,cls_logits
