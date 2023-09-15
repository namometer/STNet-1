import torch
import torch.nn as nn

from .MSA import MSANet
from .Scodecs import codecs


class TemNet(MSANet):
    def __init__(self, d_model=144, d_bq=18, d_q=18, d_v=18, n_heads=8, d_ff=72,
                 n_layers=6, cls_layers=(-1,), dropout=0.0, n_class=2,
                 kernel=5, auto=False):
        super(TemNet, self).__init__(d_model, d_bq, d_q, d_v, n_heads, d_ff,
                                       n_layers, dropout, n_class)
        self.n_class = n_class
        self.auto = auto
        self.cls_layers = [n_layers + idx for idx in cls_layers]

        self.y_ori = None

        self.name = 'TemNet'
        para = [d_model, d_bq, d_q, d_v, n_heads, d_ff, n_layers,
                dropout, n_class, kernel, auto]
        self.para_str = '(' + ', '.join([str(p) for p in para]) + ')'

    def reference(self, x, labels):
        x = self.firstLayer(x)

        x_cls = []
        x = self.dropout(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.dropout(x)
            if i + 1 in self.cls_layers:
                x_cls.append(x)
        x_cls = torch.cat(x_cls, dim=-2)

        v = x_cls.mean(dim=-2)
        y = self.fc(v)
        if labels is None:
            _, labels = torch.max(y, dim=-1)

        return y

    def forward(self, x, labels=None):
        batch_size = x.shape[0]
        x_copy = x.clone()
        y = self.reference(x_copy, labels)

        if not self.training:
            self.y_ori = y.clone()
            final_labels = torch.max(y, dim=-1)[1].long()  # self.rectify_labels(y, logit_mat)
            y = 0 * y
            for i in range(batch_size):
                y[i, final_labels[i]] = 1

        return y


class STNet(MSANet):
    def __init__(self, d_model=144, d_bq=18, d_q=18, d_v=18, n_heads=8, d_ff=72,
                 n_layers=6, cls_layers=(-1,), dropout=0.0, n_class=2,
                 kernel=5, auto=False):
        super(STNet, self).__init__(d_model, d_bq, d_q, d_v, n_heads, d_ff,
                                          n_layers, dropout, n_class)
        self.n_class = n_class
        self.auto = auto
        self.conv_ae = codecs(d_model=d_model, kernel=kernel, dropout=dropout)
        self.fusion = nn.Sequential(
            nn.Linear(2 * d_model, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
        self.cls_layers = [n_layers + idx for idx in cls_layers]

        self.y_ori = None

        self.name = 'STNet'
        para = [d_model, d_bq, d_q, d_v, n_heads, d_ff, n_layers,
                dropout, n_class, kernel, auto]
        self.para_str = '(' + ', '.join([str(p) for p in para]) + ')'

    def reference(self, x, labels):
        x = self.firstLayer(x)

        x_cls = []
        x = self.dropout(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            x = self.dropout(x)
            if i + 1 in self.cls_layers:
                x_cls.append(x)
        x_cls = torch.cat(x_cls, dim=-2)

        v = x_cls.mean(dim=-2)
        y = self.fc(v)
        if labels is None:
            _, labels = torch.max(y, dim=-1)

        return y

    def forward(self, x, labels=None):
        batch_size = x.shape[0]
        x_copy = x.clone()
        x_space = self.conv_ae(x_copy)[1]
        connected = torch.cat([x_copy, x_space], dim=2)
        x_fusion = self.fusion(connected.float())
        y = self.reference(x_fusion, labels)

        if not self.training:
            self.y_ori = y.clone()
            final_labels = torch.max(y, dim=-1)[1].long()
            y = 0 * y
            for i in range(batch_size):
                y[i, final_labels[i]] = 1

        return y
