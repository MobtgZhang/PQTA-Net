import paddle
import paddle.nn as nn

from .embedding import Embedding
from .layers import SeqAttnMatch,SelfAttnMatch
from .layers import SFU,StackedBRNN

class MReader(nn.Layer):
    def __init__(self,args):
        super(MReader, self).__init__()
        self.embedding_dim = self.embedding.embedding_dim
        self.embedding = Embedding(args.embedding_name)
        self.interactive_aligners = nn.LayerList()
        self.interactive_SFUs = nn.LayerList()
        self.self_aligners = nn.LayerList()
        self.self_SFUs = nn.LayerList()
        self.aggregate_rnns = nn.LayerList()
        doc_hidden_size = self.embedding_dim
        for i in range(args.hop):
            self.interactive_aligners.append(SeqAttnMatch(doc_hidden_size,identity=True))
            self.interactive_SFUs.append(SFU(doc_hidden_size,3*doc_hidden_size))
            # self aligner
            self.self_aligners.append(SelfAttnMatch(doc_hidden_size,identity=True,diag=False))
            self.self_SFUs.append(SFU(doc_hidden_size,doc_hidden_size*3))
            # aggregating
            self.aggregate_rnns.append(
                StackedBRNN(
                    input_size=doc_hidden_size,
                    hidden_size=args.hidden_size,
                    num_layers=1,
                    dropout_rate=args.dropout_rnn,
                    dropout_output=args.dropout_rnn_output,
                    concat_layers=False,
                    rnn_type=nn.GRU
                )
            )
















