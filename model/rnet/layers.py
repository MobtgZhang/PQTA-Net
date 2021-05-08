import paddle
import paddle.nn.functional as F
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
#------------------------R-Net---------------
class SelfAttnMatch(nn.Layer):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * x_j) for i in X
    * alpha_j = softmax(x_j * x_i)
    """
    def __init__(self, input_size,diag=True):
        super(SelfAttnMatch, self).__init__()
        weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.XavierNormal())
        bias_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.XavierNormal())
        self.linear = nn.Linear(input_size, input_size,weight_attr=weight_attr,bias_attr=bias_attr)
        self.diag = diag
    def forward(self, x):
        """
        Args:
            x: batch * len * dim
        Output:
            matched_seq: batch * len * dim
        """
        # Project vectors
        batch_size,seq_len,hidden_dim = x.shape[0],x.shape[1],x.shape[2]
        x_proj = self.linear(x.reshape(shape=[batch_size*seq_len,hidden_dim])).reshape(shape=x.shape)
        x_proj = F.relu(x_proj)
        # Compute scores
        scores = x_proj.bmm(paddle.transpose(x_proj,perm=[0,2,1]))
        if not self.diag:
            x_len = x.size(1)
            for i in range(x_len):
                scores[:, i, i] = 0
        # Normalize with softmax
        alpha = F.softmax(scores, axis=2)

        # Take weighted average
        matched_seq = alpha.bmm(x)
        return matched_seq
# ------------------------------------------------------------------------------
# Attentions
# ------------------------------------------------------------------------------

class SeqAttnMatch(nn.Layer):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """

    def __init__(self, input_size):
        super(SeqAttnMatch, self).__init__()
        weight_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.XavierNormal())
        bias_attr = paddle.framework.ParamAttr(
            initializer=paddle.nn.initializer.XavierNormal())
        self.linear = nn.Linear(input_size, input_size,weight_attr=weight_attr,bias_attr=bias_attr)

    def forward(self, x, y):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
        Output:
            matched_seq: batch * len1 * hdim
        """
        batch_size,seq_len,hid_dim = x.shape[0],x.shape[1],x.shape[2]
        # Project vectors
        x_proj = self.linear(x.reshape(shape=[batch_size*seq_len, hid_dim])).reshape(shape=x.shape)
        x_proj = F.relu(x_proj)
        batch_size, seq_len, hid_dim = y.shape[0], y.shape[1], y.shape[2]
        y_proj = self.linear(y.reshape(shape=[batch_size*seq_len, hid_dim])).reshape(shape=y.shape)
        y_proj = F.relu(y_proj)
        # Compute scores

        scores = x_proj.bmm(paddle.transpose(y_proj,perm=[0,2,1]))
        # Normalize with softmax
        alpha = F.softmax(scores, axis=2)
        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq
# ------------------------------------------------------------------------------
# Functional Units
# ------------------------------------------------------------------------------

class Gate(nn.Layer):
    """Gate Unit
    g = sigmoid(Wx)
    x = g * x
    """
    def __init__(self, input_size):
        super(Gate, self).__init__()
        weight_attr = paddle.framework.ParamAttr(initializer=paddle.nn.initializer.XavierNormal())
        self.linear = nn.Linear(input_size, input_size,weight_attr=weight_attr)
    def forward(self, x):
        """
        Args:
            x: batch * len * dim
        Output:
            res: batch * len * dim
        """
        x_proj = self.linear(x)
        gate = F.sigmoid(x)
        return x_proj * gate
class StackedBRNN(nn.Layer):
    """Stacked Bi-directional RNNs.
    Differs from standard PyTorch library in that it has the option to save
    and concat the hidden states between layers. (i.e. the output hidden size
    for each sequence input is num_layers * hidden_size).
    """

    def __init__(self, input_size, hidden_size, num_layers,
                 dropout_rate=0, dropout_output=False, rnn_type=nn.LSTM,
                 concat_layers=False):
        super(StackedBRNN, self).__init__()
        self.dropout_output = dropout_output
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.concat_layers = concat_layers
        self.rnns = nn.LayerList()
        for i in range(num_layers):
            input_size = input_size if i == 0 else 2 * hidden_size
            self.rnns.append(rnn_type(input_size, hidden_size,
                                      num_layers=1,
                                      direction="bidirectional"))

    def forward(self,x):
        """Encode either padded or non-padded sequences.
        Can choose to either handle or ignore variable length sequences.
        Always handle padding in eval.
        Args:
            x: batch * len * hdim
        Output:
            x_encoded: batch * len * hdim_encoded
        """
        output = self._forward_unpadded(x)
        return output

    def _forward_unpadded(self,x):
        """Faster encoding that ignores any padding."""
        # Transpose batch and sequence dims
        x = x.transpose(perm = [1,0,2])

        # Encode all layers
        outputs = [x]
        for i in range(self.num_layers):
            rnn_input = outputs[-1]

            # Apply dropout to hidden input
            if self.dropout_rate > 0:
                rnn_input = F.dropout(rnn_input,
                                      p=self.dropout_rate,
                                      training=self.training)
            # Forward
            rnn_output = self.rnns[i](rnn_input)[0]
            outputs.append(rnn_output)

        # Concat hidden layers
        if self.concat_layers:
            output = paddle.concat(outputs[1:], axis=2)
        else:
            output = outputs[-1]

        # Transpose back
        output = output.transpose(perm=[1,0,2])

        # Dropout on output layer
        if self.dropout_output and self.dropout_rate > 0:
            output = F.dropout(output,
                               p=self.dropout_rate,
                               training=self.training)
        return output

class FeedForwardNetwork(nn.Layer):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(FeedForwardNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        weight_attr1 = paddle.framework.ParamAttr(initializer=nn.initializer.XavierNormal())
        bias_attr1 = paddle.framework.ParamAttr(initializer=nn.initializer.XavierNormal())
        weight_attr2 = paddle.framework.ParamAttr(initializer=nn.initializer.XavierNormal())
        bias_attr2 = paddle.framework.ParamAttr(initializer=nn.initializer.XavierNormal())
        self.linear1 = nn.Linear(input_size, hidden_size,weight_attr=weight_attr1,bias_attr=bias_attr1)
        self.linear2 = nn.Linear(hidden_size, output_size,weight_attr=weight_attr2,bias_attr=bias_attr2)

    def forward(self, x):
        x_proj = F.dropout(F.relu(self.linear1(x)), p=self.dropout_rate, training=self.training)
        x_proj = self.linear2(x_proj)
        return x_proj

class NonLinearSeqAttn(nn.Layer):
    """Self attention over a sequence:
    * o_i = softmax(function(Wx_i)) for x_i in X.
    """

    def __init__(self, input_size, hidden_size):
        super(NonLinearSeqAttn, self).__init__()
        self.FFN = FeedForwardNetwork(input_size, hidden_size, 1)

    def forward(self,x):
        """
        Args:
            x: batch * len * dim
        Output:
            alpha: batch * len
        """
        scores = self.FFN(x).squeeze(2)
        alpha = F.softmax(scores)
        return alpha

class PointerNetwork(nn.Layer):
    def __init__(self, x_size, y_size, hidden_size, dropout_rate=0, cell_type=nn.GRUCell, normalize=True):
        super(PointerNetwork, self).__init__()
        self.normalize = normalize
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        weight_attr1= paddle.framework.ParamAttr(
            name="weight_attr1",
            initializer=nn.initializer.XavierNormal())
        weight_attr2 = paddle.framework.ParamAttr(
            name="weight_attr2",
            initializer=nn.initializer.XavierNormal())
        self.x_size = x_size
        self.y_size = y_size
        self.linear = nn.Linear(x_size + y_size, hidden_size, weight_attr=weight_attr1)
        self.weights = nn.Linear(hidden_size, 1,weight_attr=weight_attr2)
        self.self_attn = NonLinearSeqAttn(y_size, hidden_size)
        self.cell = cell_type(x_size, y_size)

    def init_hiddens(self,y):
        attn = self.self_attn(y)
        res = attn.unsqueeze(1).bmm(y).squeeze(1)  # [B, I]
        return res

    def pointer(self, x, state):
        x_ = paddle.fluid.layers.expand(state.unsqueeze(1),expand_times=[1, x.shape[1], 1])
        out = paddle.concat([x, x_], axis=2)
        s0 = F.tanh(self.linear(out))
        s = self.weights(s0).reshape(shape=[x.shape[0], x.shape[1]])
        a = F.softmax(s)
        res = a.unsqueeze(1).bmm(x).squeeze(1)
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                scores = F.log_softmax(s)
            else:
                # ...Otherwise 0-1 probabilities
                scores = F.softmax(s)
        else:
            scores = a.exp()
        return scores,res

    def forward(self,x,y):
        '''
        :param x: (batch_size,seq_len1,hidden_dim2)
        :param y: (batch_size,seq_len2,hidden_dim2)
        :return:
        '''
        hiddens = self.init_hiddens(y)
        start_scores,c = self.pointer(x, hiddens)
        c_ = F.dropout(c, p=self.dropout_rate, training=self.training)
        hiddens,_ = self.cell(c_, hiddens)
        end_scores,cls_logits = self.pointer(x, hiddens)
        return start_scores, end_scores,cls_logits
#------------------------End R-Net---------------


