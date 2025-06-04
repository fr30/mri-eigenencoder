import torch
import torch.nn.functional as F

from torch import nn
from torch_geometric.nn.models import GIN
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.data import Batch


class GINEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        dropout,
        norm=None,
        emb_style="none",  # ['none, 'concat', 'replace']
        num_nodes=None,
        emb_dim=128,
        norm_out="sigmoid",  # ['sigmoid', 'batch_norm', 'none']
    ):
        super().__init__()
        self.emb_style = emb_style
        self.num_nodes = num_nodes
        self.emb_dim = emb_dim

        if emb_style not in ["none", "concat", "replace"]:
            raise ValueError(f"Invalid emb_style: {emb_style}")

        if emb_style != "none":
            self.emb = nn.Embedding(num_nodes, hidden_channels)

        if emb_style == "concat":
            in_channels += hidden_channels
        elif emb_style == "replace":
            in_channels = hidden_channels

        self.gin = GIN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=emb_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
        )

        if norm_out == "batch_norm":
            self.norm_out = nn.BatchNorm1d(emb_dim * 4, affine=False)
        elif norm_out == "sigmoid":
            self.norm_out = F.sigmoid
        else:
            self.norm_out = nn.Identity()

    def forward(self, batch_graph):
        if self.emb_style != "none":
            x = self._add_emb(batch_graph.x, batch_graph.batch_size)
        else:
            x = batch_graph.x

        edge_index = batch_graph.edge_index
        x = self.gin(x, edge_index)
        x = self.norm_out(x)
        x = global_mean_pool(x, batch_graph.batch)

        return x

    def _add_emb(self, x, batch_size):
        nids = torch.arange(self.num_nodes).repeat(batch_size).to(x.device)
        emb = self.emb(nids)

        if self.emb_style == "concat":
            x = torch.cat([x, emb], dim=1)
        elif self.emb_style == "replace":
            x = emb

        return x


class Classifier(nn.Module):
    def __init__(
        self,
        encoder,
        num_classes,
        linear=False,
    ):
        super().__init__()
        self.encoder = encoder
        if linear:
            self.cls_head = nn.Linear(encoder.emb_dim, num_classes)
        else:
            self.cls_head = nn.Sequential(
                nn.Linear(encoder.emb_dim, encoder.emb_dim * 4),
                nn.ReLU(),
                nn.Linear(encoder.emb_dim * 4, encoder.emb_dim * 4),
                nn.ReLU(),
                nn.Linear(encoder.emb_dim * 4, num_classes),
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.cls_head(x)
        return x.squeeze()


class GINEncoderWithProjector(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        dropout,
        norm=None,
        emb_style="none",  # ['none, 'concat', 'replace']
        num_nodes=None,
        emb_dim=128,
        norm_out="batch",  # ['sigmoid', 'batch', 'none']
        enc_norm_out="sigmoid",  # ['sigmoid', 'batch', 'none']
    ):
        super().__init__()
        self.encoder = GINEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            emb_style=emb_style,
            num_nodes=num_nodes,
            emb_dim=emb_dim,
            norm_out=enc_norm_out,
        )

        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.BatchNorm1d(emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim * 4),
            nn.BatchNorm1d(emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim * 4),
            nn.BatchNorm1d(emb_dim * 4),
            nn.ReLU(),
        )
        if norm_out == "batch":
            self.norm_out = nn.BatchNorm1d(emb_dim * 4, affine=False)
        elif norm_out == "sigmoid":
            self.norm_out = F.sigmoid
        else:
            self.norm_out = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        x = self.norm_out(x)

        return x


class BarlowTwinsGIN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        dropout,
        norm=None,
        emb_style="none",  # ['none, 'concat', 'replace']
        num_nodes=None,
        emb_dim=128,
        norm_out="batch",  # ['sigmoid', 'batch', 'none']
    ):
        super().__init__()

        self.model = GINEncoderWithProjector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            emb_style=emb_style,
            num_nodes=num_nodes,
            emb_dim=emb_dim,
            norm_out=norm_out,
            enc_norm_out="none",
        )

    def forward(self, x1, x2):
        return self.model(x1), self.model(x2)

    @property
    def encoder(self):
        return self.model.encoder


class SFCNEncoder(nn.Module):
    def __init__(
        self,
        channel_number=[28, 58, 128, 256, 512],
        emb_dim=128,
        norm_out="sigmoid",  # ['sigmoid', 'batch', 'none']
    ):
        super().__init__()
        self.emb_dim = emb_dim
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
        self.feature_extractor.add_module("avgpool", nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.dropout = nn.Dropout(0.5)
        # Change to self.linear = nn.Linear(...) when training next series of checkpoints
        self.linear = nn.Sequential(nn.Linear(channel_number[-1], emb_dim))

        if norm_out == "batch":
            self.norm_out = nn.BatchNorm1d(emb_dim, affine=False)
        elif norm_out == "sigmoid":
            self.norm_out = F.sigmoid
        else:
            self.norm_out = nn.Identity()

    def forward(self, x):
        x = self.feature_extractor(x).reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.norm_out(x)
        return x

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
                nn.ReLU(),
                nn.MaxPool3d(3, stride=maxpool_stride),
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


class DinoEncoder(nn.Module):
    def __init__(self, size="s"):  # ['s', 'b', 'l', 'g']
        super().__init__()
        # self.feature_extractor = torch.hub.load(
        #     "facebookresearch/dinov2", "dinov2_vitb14_reg"
        # )
        self.feature_extractor = torch.hub.load(
            "facebookresearch/dinov2", f"dinov2_vit{size}14_reg"
        )
        self.emb_dim = self.feature_extractor.embed_dim

    def forward(self, x):
        x = self._cube_to_slides(x)
        x = self.feature_extractor(x)
        return x

    @staticmethod
    def _cube_to_slides(x):
        x = x.unbind(dim=-1)
        x = torch.concat(x, dim=-1)
        x = x.unbind(dim=1)
        x = torch.concat(x, dim=-1)
        _, h, w = x.shape
        x = x[:, : h // 14 * 14, : w // 14 * 14]
        x = x.unsqueeze(1).expand(-1, 3, -1, -1)
        return x


class SFCNClassifier(nn.Module):
    def __init__(
        self,
        output_dim,
        encoder=None,
        channel_number=[28, 58, 128, 256, 512],
        emb_dim=128,
        hidden_dim=1024,
    ):
        super().__init__()

        if encoder is None:
            self.encoder = SFCNEncoder(channel_number=channel_number, emb_dim=emb_dim)
        else:
            self.encoder = encoder

        self.linear = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear(x)
        return x.reshape(-1)


class SFCNEncoderWithProjector(nn.Module):
    def __init__(
        self,
        channel_number=[28, 58, 128, 256, 512],
        emb_dim=128,
        norm_out="batch",  # ['sigmoid', 'batch', 'none']
        enc_norm_out="sigmoid",  # ['sigmoid', 'batch_norm', 'none']
    ):
        super().__init__()
        self.encoder = SFCNEncoder(
            channel_number=channel_number, emb_dim=emb_dim, norm_out=enc_norm_out
        )
        self.projector = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.BatchNorm1d(emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim * 4),
            nn.BatchNorm1d(emb_dim * 4),
            nn.ReLU(),
            nn.Linear(emb_dim * 4, emb_dim * 4),
            nn.BatchNorm1d(emb_dim * 4),
            nn.ReLU(),
        )

        if norm_out == "batch_norm":
            self.norm_out = nn.BatchNorm1d(emb_dim * 4, affine=False)
        elif norm_out == "sigmoid":
            self.norm_out = F.sigmoid
        else:
            self.norm_out = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        x = self.projector(x)
        x = self.norm_out(x)
        # Try replacing batchnorm with sigmoid
        # Add some noise to input
        # Run for fewer epochs
        return x


class SimpleCNN(nn.Module):
    def __init__(self, in_channels=64, HIDDEN=200, out_channels=64, sample_time=2):
        super(SimpleCNN, self).__init__()

        self.cnn_list = []
        self.bn_list = []

        self.dim = out_channels

        self.cnn_list.append(
            nn.Conv1d(
                in_channels, HIDDEN, kernel_size=1, stride=1, padding=0, bias=True
            )
        )
        self.bn_list.append(nn.BatchNorm1d(HIDDEN))

        self.cnn_list.append(
            nn.Conv1d(
                HIDDEN, HIDDEN, kernel_size=sample_time, stride=1, padding=0, bias=True
            )
        )
        self.bn_list.append(nn.BatchNorm1d(HIDDEN))

        self.cnn_list = nn.ModuleList(self.cnn_list)
        self.bn_list = nn.ModuleList(self.bn_list)

        self.cnn_final = nn.Conv1d(
            HIDDEN, out_channels, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x):

        for i in range(0, len(self.cnn_list)):
            x = self.cnn_list[i](x)
            x = torch.relu(x)
            x = self.bn_list[i](x)

        x = self.cnn_final(x)
        x = torch.sigmoid(x)

        return x


class HFMCAGIN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_layers,
        dropout,
        nviews,
        norm=None,
        emb_style="none",  # ['none, 'concat', 'replace']
        num_nodes=None,
        emb_dim=128,
    ):
        super().__init__()
        self.backbone = GINEncoderWithProjector(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            emb_dim=emb_dim,
            dropout=dropout,
            norm=norm,
            emb_style=emb_style,
            num_nodes=num_nodes,
            norm_out="batch",
            enc_norm_out="batch",
        )
        self.final_net = SimpleCNN(4 * emb_dim, 4 * emb_dim, emb_dim, nviews)
        self.nviews = nviews

    def forward(self, x):
        bsize = x.batch_size // self.nviews
        y = self.backbone(x)
        y = y.squeeze(-1).squeeze(-1)
        y = torch.stack([y[bsize * k : bsize * (k + 1)] for k in range(0, self.nviews)])
        y = y.permute(1, 2, 0)

        y_dash = self.final_net(y).squeeze(-1)

        return y, y_dash

    @property
    def encoder(self):
        return self.backbone.encoder
