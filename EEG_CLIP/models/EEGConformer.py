import torch
import torch.nn as nn
from torchsummary import summary


class ShallowNet(nn.Module):
    def __init__(self, config):
        super(ShallowNet, self).__init__()
        self.temporal_conv = nn.Conv2d(in_channels=config["in_channels"], out_channels=config["channel_size"],
                                       kernel_size=(1, config["temporal_conv_size"]), stride=(1, 1))
        self.spatial_conv = nn.Conv2d(in_channels=config["channel_size"], out_channels=config["channel_size"],
                                      kernel_size=(config["spatial_conv_size"], 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(config["channel_size"])
        self.elu = nn.ELU()
        self.avgpool = nn.AvgPool2d((1, config["avg_kernel"]), (1, config["avg_stride"]))
        self.dr = nn.Dropout(config["dropout"])

    def forward(self, inputs):
        x = self.temporal_conv(inputs)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.avgpool(x)
        x = self.dr(x)
        return x


class EEGConformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shallow_net = ShallowNet(config)
        self.multi_head_attention = nn.MultiheadAttention(config["channel_size"], config["num_head"],
                                                          dropout=config["dropout"], batch_first=True)
        self.n_layers = config["num_attention"]
        self.projection = nn.Linear(16 * 40, 100)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, inputs):
        x = self.shallow_net(inputs)
        x = x.permute(0, 2, 3, 1).squeeze(1)

        for _ in range(self.n_layers):
            residual = x
            x, _ = self.multi_head_attention(x, x, x)
            x += residual
        x = self.flatten(x)
        x = self.relu(self.projection(x))
        return x


if __name__ == "__main__":
    configs = {
        "in_channels": 1,
        "channel_size": 40,
        "temporal_conv_size": 25,
        "spatial_conv_size": 60,
        "avg_kernel": 25,
        "avg_stride": 5,
        "dropout": 0.5,
        "num_head": 10,
        "num_attention": 6
    }
    temp = torch.randn(1, 1, 60, 128)
    net = EEGConformer(configs)
    print(summary(net, (1, 60, 128), device="cpu"))