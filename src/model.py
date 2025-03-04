from torch import nn
from torch_geometric.nn.models import GIN


class GINClassifier(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, num_layers, out_channels, dropout, norm=None
    ):
        super(GINClassifier, self).__init__()
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


class SFCN(nn.Module):
    def __init__(
        self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=40, dropout=True
    ):
        super(SFCN, self).__init__()
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
        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]
        self.classifier.add_module("average_pool", nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module("dropout", nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module(
            "conv_%d" % i, nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1)
        )

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

    def forward(self, x):
        x_f = self.feature_extractor(x)
        out = self.classifier(x_f)
        return out.reshape(-1)
