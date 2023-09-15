import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, d_in, d_out, kernel, stride, padding, with_shortcut=False, alpha=0.2, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.with_shortcut = with_shortcut
        self.conv = nn.Conv1d(d_in, d_out, kernel_size=(kernel,), stride=(stride,), padding=(padding,), bias=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, shortcut = x
        y = self.dropout(self.leaky_relu(self.conv(x)))
        if self.with_shortcut:
            shortcut.append(y)
            return y, shortcut
        else:
            return y, shortcut


class DecoderLayer(nn.Module):
    def __init__(self, d_in, d_out, kernel, k, padding, with_shortcut=False, alpha=0.2, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.with_shortcut = with_shortcut
        self.up_sample = nn.Upsample(scale_factor=k, mode='linear', align_corners=False)
        self.conv = nn.Conv1d(d_in, d_out, kernel_size=(kernel,), stride=(1,), padding=(padding,), bias=True)
        self.pool = nn.AvgPool1d(kernel_size=(3,), stride=(1,), padding=(1,))
        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x, shortcut = x
        if self.with_shortcut:
            y = self.dropout(self.leaky_relu(self.pool(self.conv(self.up_sample(x + shortcut.pop())))))
        else:
            y = self.dropout(self.leaky_relu(self.pool(self.conv(self.up_sample(x)))))
        return y, shortcut


class ConvEndLayer(nn.Module):
    def __init__(self, d_in, d_out, kernel, padding, with_shortcut=False):
        super(ConvEndLayer, self).__init__()
        self.with_shortcut = with_shortcut
        self.conv = nn.Conv1d(d_in, d_out, kernel_size=(kernel,), stride=(1,), padding=(padding,), bias=True)

    def forward(self, x):
        x, shortcut = x
        if self.with_shortcut:
            y = self.conv(x + shortcut.pop())
        else:
            y = self.conv(x)
        return y


class codecs(nn.Module):
    def __init__(self, d_model=144, kernel=5, dropout=0.1):
        super(codecs, self).__init__()
        pad = int(kernel / 2)
        self.encoder = nn.Sequential(
            EncoderLayer(d_model, d_model, kernel, 1, pad, False, dropout=dropout),
            EncoderLayer(d_model, d_model, kernel, 2, pad, False, dropout=dropout),
            EncoderLayer(d_model, int(d_model / 2), kernel, 1, pad, False, dropout=dropout),
            EncoderLayer(int(d_model / 2), int(d_model / 2), kernel, 2, pad, False, dropout=dropout),
            EncoderLayer(int(d_model / 2), int(d_model / 4), kernel, 1, pad, False, dropout=dropout)
        )
        self.decoder = nn.Sequential(
            DecoderLayer(int(d_model / 4), int(d_model / 2), kernel, 1, pad, False, dropout=dropout),
            DecoderLayer(int(d_model / 2), int(d_model / 2), kernel, 2, pad, False, dropout=dropout),
            DecoderLayer(int(d_model / 2), d_model, kernel, 1, pad, False, dropout=dropout),
            DecoderLayer(d_model, d_model, kernel, 2, pad, False, dropout=dropout),
            DecoderLayer(d_model, d_model, kernel, 1, pad, False, dropout=dropout),
            ConvEndLayer(d_model, d_model, kernel, pad, False)
        )

        self.name = 'Scodecs'
        para = [d_model, kernel]
        self.para_str = '(' + ', '.join([str(p) for p in para]) + ')'

    def forward(self, x):
        """
        :param x: [batch_size, seq_len, d_model]
        :return:
        """
        x = x.transpose(-1, -2)
        feature = self.encoder((x, []))
        recover = self.decoder(feature)
        return feature[0].transpose(-1, -2), recover.transpose(-1, -2)
