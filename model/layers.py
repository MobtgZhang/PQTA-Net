import paddle
import paddle.nn as nn
import paddle.nn.functional as F

#------------------------Common Layers---------------
class SelfAttnMatch(nn.Layer):
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * x_j) for i in X
    * alpha_j = softmax(x_j * x_i)
    """
    def __init__(self, input_size,identity=False,diag=True):
        super(SelfAttnMatch, self).__init__()
        if not identity:
            weight_attr = paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal())
            bias_attr = paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal())
            self.linear = nn.Linear(input_size, input_size,weight_attr=weight_attr,bias_attr=bias_attr)
        else:
            self.linear = None
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
        if self.linear:
            x_proj = self.linear(x.reshape(shape=[batch_size*seq_len,hidden_dim])).reshape(shape=x.shape)
            x_proj = F.relu(x_proj)
        else:
            x_proj = x
        # Compute scores
        scores = x_proj.bmm(paddle.transpose(x_proj,perm=[0,2,1]))
        if not self.diag:
            x_len = x.shape[1]
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

    def __init__(self, input_size,identity=False):
        super(SeqAttnMatch, self).__init__()
        if not identity:
            weight_attr = paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal())
            bias_attr = paddle.framework.ParamAttr(
                initializer=paddle.nn.initializer.XavierNormal())
            self.linear = nn.Linear(input_size, input_size,weight_attr=weight_attr,bias_attr=bias_attr)
        else:
            self.linear = None
    def forward(self,x,y):
        """
        Args:
            x: batch * len1 * hdim
            y: batch * len2 * hdim
        Output:
            matched_seq: batch * len1 * hdim
        """
        batch_size,seq_len,hid_dim = x.shape[0],x.shape[1],x.shape[2]
        # Project vectors
        if self.linear:
            x_proj = self.linear(x.reshape(shape=[batch_size*seq_len, hid_dim])).reshape(shape=x.shape)
            x_proj = F.relu(x_proj)
            batch_size, seq_len, hid_dim = y.shape[0], y.shape[1], y.shape[2]
            y_proj = self.linear(y.reshape(shape=[batch_size*seq_len, hid_dim])).reshape(shape=y.shape)
            y_proj = F.relu(y_proj)
        else:
            x_proj = x
            y_proj = y
        # Compute scores

        scores = x_proj.bmm(paddle.transpose(y_proj,perm=[0,2,1]))
        # Normalize with softmax
        alpha = F.softmax(scores, axis=2)
        # Take weighted average
        matched_seq = alpha.bmm(y)
        return matched_seq
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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size =output_size
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
#------------------------End Common Layers-----------

#------------------------R-Net---------------
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


#------------------------M-Reader---------------

class SFU(nn.Layer):
    """Semantic Fusion Unit
    The ouput vector is expected to not only retrieve correlative information from fusion vectors,
    but also retain partly unchange as the input vector
    """
    def __init__(self, input_size, fusion_size):
        super(SFU, self).__init__()
        weight_attr_r = paddle.framework.ParamAttr(initializer=nn.initializer.XavierNormal())
        bias_attr_r = paddle.framework.ParamAttr(initializer=nn.initializer.XavierNormal())
        weight_attr_g = paddle.framework.ParamAttr(initializer=nn.initializer.XavierNormal())
        bias_attr_g = paddle.framework.ParamAttr(initializer=nn.initializer.XavierNormal())
        self.linear_r = nn.Linear(input_size + fusion_size, input_size,weight_attr=weight_attr_r,bias_attr=bias_attr_r)
        self.linear_g = nn.Linear(input_size + fusion_size, input_size,weight_attr=weight_attr_g,bias_attr=bias_attr_g)

    def forward(self, x, fusions):
        r_f = paddle.concat([x, fusions], axis=2)
        r = F.tanh(self.linear_r(r_f))
        g = F.sigmoid(self.linear_g(r_f))
        o = g * r + (1-g) * x
        return o

class MemoryAnsPointer(nn.Layer):
    def __init__(self, x_size, y_size,hidden_size, hop=1, dropout_rate=0, normalize=True):
        super(MemoryAnsPointer, self).__init__()
        self.normalize = normalize
        self.hidden_size = hidden_size
        self.hop = hop
        self.x_size = x_size
        self.y_size = y_size
        self.dropout_rate = dropout_rate
        self.ques_encoder = nn.GRU(input_size=y_size,hidden_size=x_size,num_layers=1,
                                   direction='forward',time_major=False)
        self.FFNs_start = nn.LayerList()
        self.SFUs_start = nn.LayerList()
        self.FFNs_end = nn.LayerList()
        self.SFUs_end = nn.LayerList()
        for i in range(self.hop):
            self.FFNs_start.append(FeedForwardNetwork(3*x_size, hidden_size, 1, dropout_rate))
            self.SFUs_start.append(SFU(x_size, x_size))
            self.FFNs_end.append(FeedForwardNetwork(3*x_size, hidden_size, 1, dropout_rate))
            self.SFUs_end.append(SFU(x_size, x_size))

    def forward(self, x, y):
        _,z_s = self.ques_encoder(y)
        z_s = paddle.fluid.layers.transpose(z_s, perm=[1, 0, 2]) # [B,1,I]
        for i in range(self.hop):
            z_s_ = paddle.fluid.layers.expand(z_s,expand_times=[1, x.shape[1], 1])  # [B, S, I]
            s = self.FFNs_start[i](paddle.concat([x, z_s_, x * z_s_], axis=2)).squeeze(2)
            p_s = F.softmax(s, axis=1)  # [B, S]
            u_s = p_s.unsqueeze(1).bmm(x)  # [B, 1, I]
            z_e = self.SFUs_start[i](z_s, u_s)  # [B, 1, I]
            z_s_ = paddle.fluid.layers.expand(z_s,expand_times=[1, x.shape[1], 1])  # [B, S, I]
            e = self.FFNs_end[i](paddle.concat([x, z_s_, x * z_s_], axis=2)).squeeze(2)
            p_e = F.softmax(e, axis=1)  # [B, S]
            u_e = p_e.unsqueeze(1).bmm(x)  # [B, 1, I]
            z_s = self.SFUs_end[i](z_e, u_e)
        if self.normalize:
            if self.training:
                # In training we output log-softmax for NLL
                p_s = F.log_softmax(s, axis=1)  # [B, S]
                p_e = F.log_softmax(e, axis=1)  # [B, S]
            else:
                # ...Otherwise 0-1 probabilities
                p_s = F.softmax(s, axis=1)  # [B, S]
                p_e = F.softmax(e, axis=1)  # [B, S]
        else:
            p_s = s.exp()
            p_e = e.exp()
        return p_s, p_e,z_s
#------------------------End M-Reader---------------
#------------------------End PQTA-Net---------------
class AggrateBiRNN(nn.Layer):
    def __init__(self):
        super(AggrateBiRNN, self).__init__()
    def forward(self,puts_emb):
        pass
#------------------------End PQTA-Net---------------
