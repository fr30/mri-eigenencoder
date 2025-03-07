import torch

from torch import nn
from torch_geometric.nn.models import GIN


class GINEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        out_channels,
        dropout,
        norm=None,
        emb_style="none",  # ['none, 'concat', 'replace']
        num_nodes=None,
        emb_size=128,
    ):
        super().__init__()
        self.emb_style = emb_style
        self.num_nodes = num_nodes

        if emb_style not in ["none", "concat", "replace"]:
            raise ValueError(f"Invalid emb_style: {emb_style}")

        if emb_style != "none":
            self.emb = nn.Embedding(num_nodes, emb_size)

        if emb_style == "concat":
            in_channels += emb_size
        elif emb_style == "replace":
            in_channels = emb_size

        self.gin = GIN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
        )

    def forward(self, batch_graph):
        if self.emb_style != "none":
            x = self._add_emb(batch_graph.x, batch_graph.batch_size)
        else:
            x = batch_graph.x

        edge_index = batch_graph.edge_index
        x = self.gin(x, edge_index)
        x = x.reshape(batch_graph.batch_size, -1, x.shape[1])
        x = x.mean(dim=1)
        return x.reshape(-1)

    def _add_emb(self, x, batch_size):
        nids = torch.arange(self.num_nodes).repeat(batch_size).to(x.device)
        emb = self.emb(nids)

        if self.emb_style == "concat":
            x = torch.cat([x, emb], dim=1)
        elif self.emb_style == "replace":
            x = emb

        return x


class SFCNEncoder(nn.Module):
    def __init__(
        self,
        channel_number=[32, 64, 128, 256, 256, 64],
    ):
        super().__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()

        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(
                        in_channel, out_channel, maxpool=True, kernel_size=3, padding=1
                    ),
                )
            else:
                self.feature_extractor.add_module(
                    "conv_%d" % i,
                    self.conv_layer(
                        in_channel, out_channel, maxpool=False, kernel_size=1, padding=0
                    ),
                )

        avg_shape = [5, 6, 5]
        self.feature_extractor.add_module("avgpool", nn.AvgPool3d(avg_shape))

    def forward(self, x):
        bsize = x.shape[0]
        x = self.feature_extractor(x)
        return x.reshape(bsize, -1)

    @staticmethod
    def conv_layer(
        in_channel,
        out_channel,
        maxpool=True,
        kernel_size=3,
        padding=0,
        maxpool_stride=2,
    ):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channel, out_channel, padding=padding, kernel_size=kernel_size
                ),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(
                    in_channel, out_channel, padding=padding, kernel_size=kernel_size
                ),
                nn.BatchNorm3d(out_channel),
                nn.ReLU(),
            )
        return layer


class SFCNClassifier(nn.Module):
    def __init__(
        self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True
    ):
        super().__init__()
        self.encoder = SFCNEncoder(channel_number=channel_number)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(channel_number[-1], output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x.reshape(-1)
