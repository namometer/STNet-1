import torch
import numpy as np
import torch.nn as nn


class FeedForwardNet(nn.Module):
    def __init__(self, d_model=144, d_ff=256, dropout=0.0):
        super(FeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.layerNorm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        return self.layerNorm(output + residual)


class STDotProductAttention(nn.Module):
    def __init__(self, d_model=144, d_bq=18, d_q=18, d_v=18, dropout=0.0):
        super(STDotProductAttention, self).__init__()
        self.b_Wv = nn.Linear(d_model, d_v, bias=False)

        self.Wq = nn.Linear(d_model, d_q, bias=False)
        self.Wk = nn.Linear(d_model, d_q, bias=False)
        self.Wq2 = nn.Linear(d_model, d_q, bias=False)
        self.Wk2 = nn.Linear(d_model, d_q, bias=False)
        self.Wv = nn.Linear(d_model, d_v, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, scale=None, attn_mask=False):
        baseline = x.mean(dim=-2, keepdim=True)
        base_v = self.b_Wv(baseline.float())
        dx = x - baseline
        dx = dx.float()

        Q = self.Wq(dx)
        K = self.Wk(dx)
        Q2 = self.Wq(dx)
        K2 = self.Wk(dx)
        V = self.Wv(dx)
        t_scores = torch.bmm(Q, K.transpose(-1, -2))
        s_scores = torch.bmm(Q2.transpose(-1, -2), K2) / Q.shape[-2]

        if scale:
            t_scores /= scale

        if attn_mask is True:
            t = torch.triu(torch.ones(t_scores.size())).to(x.device)
            t_scores.masked_fill_(t.bool(), -1e9)
        t_attn = self.softmax(t_scores)
        s_attn = self.softmax(s_scores)
        t_attn = self.dropout(t_attn)
        s_attn = self.dropout(s_attn)
        context = t_attn.bmm(V) + base_v
        context = context.bmm(s_attn.transpose(-1, -2))
        return context


class SpTmMultiHeadAttention(nn.Module):
    def __init__(self, d_model=144, d_bq=18, d_q=18, d_v=18, n_heads=8, dropout=0.0):
        super(SpTmMultiHeadAttention, self).__init__()
        self.heads = nn.ModuleList([STDotProductAttention(d_model, d_bq, d_q, d_v, dropout) for _ in range(n_heads)])

        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.layerNorm = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=False):
        residual = x.float()
        len_seq = x.size(1)

        scale = np.sqrt(len_seq)
        contexts = []
        for head in self.heads:
            contexts.append(head(x, scale, attn_mask))

        context = torch.cat(contexts, dim=-1)

        output = self.fc(context)

        output = self.dropout(output)

        output = self.layerNorm(output + residual)

        return output


class STEncoderLayer(nn.Module):
    def __init__(self, d_model=144, d_bq=18, d_q=18, d_v=18, n_heads=8, d_ff=256, dropout=0.0):
        super(STEncoderLayer, self).__init__()
        self.MSA = SpTmMultiHeadAttention(d_model, d_bq, d_q, d_v, n_heads, dropout)
        self.posFFN = FeedForwardNet(d_model, d_ff, dropout)

    def forward(self, x, attn_mask=False):
        output = self.MSA(x, attn_mask)
        output = self.posFFN(output)
        return output


class MSANet(nn.Module):
    def __init__(self, d_model=144, d_bq=18, d_q=18, d_v=18, n_heads=8, d_ff=256, n_layers=6, dropout=0.0, n_class=2):
        super(MSANet, self).__init__()
        self.n_class = n_class
        self.firstLayer = STEncoderLayer(d_model, d_bq, d_q, d_v, n_heads, d_ff, dropout)
        self.layers = nn.ModuleList(
            [STEncoderLayer(d_model, d_bq, d_q, d_v, n_heads, d_ff, dropout) for _ in range(n_layers - 1)])
        self.dropout = nn.Dropout(max(0.1, dropout))
        self.dropout_att = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, n_class, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.name = 'MSA'
        para = [d_model, d_bq, d_q, d_v, n_heads, d_ff, n_layers, dropout, n_class]
        self.para_str = '(' + ', '.join([str(p) for p in para]) + ')'

    def forward(self, x):
        x = self.firstLayer(x, True)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x)
            x = self.dropout(x)
        y = self.fc(x[:, -1, :])
        if not self.training:
            y = self.softmax(y)
        return y
