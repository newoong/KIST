import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torchsummary import summary


def build_down_block(in_ch, out_ch):
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding="same"),
        nn.ReLU(),
        nn.BatchNorm2d(out_ch),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    return block


class AudioEncoder(nn.Module):
    def __init__(self, input_shape, n_layer, channels):
        super().__init__()
        self.input_shape = input_shape
        self.channels = channels
        self.n_layer = n_layer
        self.in_conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels[0], kernel_size=7, stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(channels[0])
        )

        blocks = []
        in_ch = channels[0]
        for i in range(n_layer):
            blocks.append(build_down_block(in_ch, channels[i]))
            in_ch = channels[i]
        self.down_blocks = nn.Sequential(*blocks)

    def forward(self, inputs):
        x = self.in_conv_block(inputs)
        x = self.down_blocks(x)
        return x


def build_up_block(in_ch, out_ch):
    block = nn.Sequential(
        nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding="same"),
        nn.ReLU(),
        nn.ConvTranspose2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, output_padding=1,
                           stride=2)
    )
    return block


class AudioDecoder(nn.Module):
    def __init__(self, input_shape, n_layer, channels):
        super().__init__()
        channels.reverse()
        self.input_shape = input_shape
        self.n_layer = n_layer
        self.channel = channels[0]

        blocks = []
        in_ch = channels[0]
        for i in range(n_layer):
            blocks.append(build_up_block(in_ch, channels[i]))
            in_ch = channels[i]
        self.up_blocks = nn.Sequential(*blocks)
        self.out_conv = nn.Conv2d(in_channels=channels[-1], out_channels=1, kernel_size=1, stride=1, padding="same")

    def forward(self, inputs):
        if inputs.dim()<=2:
            inputs = inputs.view(-1, self.channel, (self.input_shape[1] // (2**self.n_layer)),
                   (self.input_shape[2] // (2**self.n_layer)))
        x = self.up_blocks(inputs)
        return torch.sigmoid(self.out_conv(x))


class AudioAE(nn.Module):
    def __init__(self, input_shape, n_layer, channels):
        super().__init__()
        self.input_shape = input_shape
        self.channels = channels
        self.n_layer = n_layer
        self.encoder = AudioEncoder(input_shape, n_layer, channels)
        self.decoder = AudioDecoder(input_shape, n_layer, channels)

    def forward(self, inputs):
        z = self.encoder(inputs)
        out = self.decoder(z)
        return out


def build_down_block_1d(in_ch, out_ch):
    block = nn.Sequential(
        nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding="same"),
        nn.ReLU(),
        nn.BatchNorm1d(out_ch),
        nn.MaxPool1d(kernel_size=2, stride=2)
    )
    return block


class AudioEncoder1D(nn.Module):
    def __init__(self, audio_len, n_mels, n_layer, channels, latent_dim):
        super().__init__()
        self.in_conv_block = nn.Sequential(
            nn.Conv1d(in_channels=n_mels, out_channels=channels[0], kernel_size=7, stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm1d(channels[0])
        )

        blocks = []
        in_ch = channels[0]
        for i in range(n_layer):
            blocks.append(build_down_block_1d(in_ch, channels[i]))
            in_ch = channels[i]
        self.down_blocks = nn.Sequential(*blocks)
        self.fc = nn.Linear(in_features=(channels[-1]) * (audio_len // (2**n_layer)), out_features=latent_dim)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.in_conv_block(inputs)
        x = self.down_blocks(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return self.relu(x)


def build_up_block_1d(in_ch, out_ch):
    block = nn.Sequential(
        nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding="same"),
        nn.ReLU(),
        nn.ConvTranspose1d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, output_padding=1,
                           stride=2)
    )
    return block


class AudioDecoder1D(nn.Module):
    def __init__(self, audio_len, n_mels, n_layer, channels, latent_dim):
        super().__init__()
        channels.reverse()

        self.n_layer = n_layer
        self.channel = channels[0]
        self.audio_len = audio_len

        self.fc = nn.Linear(in_features=latent_dim,
                            out_features=(audio_len // (2**n_layer)) * channels[0])
        self.relu = nn.ReLU()

        blocks = []
        in_ch = channels[0]
        for i in range(n_layer):
            blocks.append(build_up_block_1d(in_ch, channels[i]))
            in_ch = channels[i]
        self.up_blocks = nn.Sequential(*blocks)
        self.out_conv = nn.Conv1d(in_channels=channels[-1], out_channels=n_mels,
                                  kernel_size=1, stride=1, padding="same")

    def forward(self, inputs):
        x = self.relu(self.fc(inputs))
        x = x.view(-1, self.channel, (self.audio_len // (2**self.n_layer)))
        x = self.up_blocks(x)
        return torch.sigmoid(self.out_conv(x))


class AudioAE1D(nn.Module):
    def __init__(self, audio_len, n_mels, n_layer, channels, latent_dim):
        super().__init__()

        self.encoder = AudioEncoder1D(audio_len, n_mels, n_layer, channels, latent_dim)
        self.decoder = AudioDecoder1D(audio_len, n_mels, n_layer, channels, latent_dim)

    def forward(self, inputs):
        z = self.encoder(inputs)
        out = self.decoder(z)
        return out


if __name__ == "__main__":
    model = AudioAE((1, 40, 200), 3, [15, 40, 60], 100)
    print(summary(model, (1, 40, 200), device="cpu"))
