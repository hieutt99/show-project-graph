import torch.nn as nn
from .utils import characters


def _cnn_cfg():
    return [
        # filters, kernel_size, stride, padding
        [64,       3,      1, 1],
        [128,      2,      2, 0],  # maxpool
        [64,       1,      1, 0],
        [128,      3,      1, 1],
        'res',
        [64,       1,      1, 0],
        [128,      3,      1, 1],
        'res',
        [256,      2,      2, 0],  # maxpool
        [128,      1,      1, 0],
        [256,      3,      1, 1],
        'res',
        [128,      1,      1, 0],
        [256,      3,      1, 1],
        'res',
        [512, (2, 1), (2, 1), 0],  # maxpool
        [256,      1,      1, 0],
        [512,      3,      1, 1],
        'res',
        [256,      1,      1, 0],
        [512,      3,      1, 1],
        'res',
        [512, (2, 1), (2, 1), 0],  # maxpool
        [512,      1,      1, 0],
        [512,      2,      1, 0]
    ]


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        cnn_cfg = _cnn_cfg()

        cnn_layers = []
        self.res_connects = []

        nf = 1
        for i, cfg in enumerate(cnn_cfg):
            if cfg == 'res':
                self.res_connects.append(i)
                cnn_layers.append(nn.Sequential())
            else:  # conv + relu
                f, k, s, p = cfg
                cnn_layers.append(nn.Sequential(
                    nn.Conv2d(nf, f, k, s, p),
                    nn.ReLU(False)
                ))
                nf = f

        self.cnn_layers = nn.Sequential(*cnn_layers)
        self.lstm = nn.LSTM(512, 256, num_layers=2, bidirectional=True)
        self.final_mapping = nn.Linear(512, len(characters) + 1)

    def forward(self, x, lengths):
        b = len(x)

        cache = None
        for i, layer in enumerate(self.cnn_layers):
            if i in self.res_connects:
                x = x + cache
            else:
                x = layer(x)
                if i + 3 in self.res_connects:
                    cache = x

        x = x.squeeze(2)
        x = x.permute(2, 0, 1).contiguous()

        lengths = lengths // 4 - 1

        x = nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=True)
        x = self.lstm(x)[0]
        x = nn.utils.rnn.pad_packed_sequence(x)[0]

        x = x.reshape((-1, 512))
        x = self.final_mapping(x)
        x = x.reshape((-1, b, len(characters) + 1))

        x = nn.functional.log_softmax(x, dim=2)
        return x, lengths
