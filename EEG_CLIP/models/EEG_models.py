import torch
import torch.nn as nn
from torchsummary import summary


class ShallowEncoderNet(nn.Module):
    def __init__(self, config):
        super(ShallowEncoderNet, self).__init__()
        self.temporal_conv = nn.Conv2d(in_channels=config["in_channels"], out_channels=config["channel_size"],
                                       kernel_size=(1, config["temporal_conv_size"]), stride=(1, 1))
        self.spatial_conv = nn.Conv2d(in_channels=config["channel_size"], out_channels=config["channel_size"],
                                      kernel_size=(config["spatial_conv_size"], 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(config["channel_size"])
        self.elu = nn.ELU()
        self.avgpool = nn.AvgPool2d((1, config["avg_kernel"]), (1, config["avg_stride"]))
        self.dr = nn.Dropout(config["dropout"])
        self.enhance_conv = nn.Conv2d(config["channel_size"], config["channel_size"], (1, 1), stride=(1, 1)) #could enhance fiting ability slightly

    def forward(self, inputs):
        x = self.temporal_conv(inputs)
        x = self.spatial_conv(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.avgpool(x)
        x = self.dr(x)
        x = self.enhance_conv(x)
        return x
    
    
class  TransformerEncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(config["channel_size"], config["num_head"],
                                                          dropout=config["dropout"], batch_first=True)
        self.feed_foward = nn.Sequential(nn.Linear(config["channel_size"], config["expansion"] * config["channel_size"]),
                                         nn.GELU(),
                                         nn.Dropout(config["dropout"]),
                                         nn.Linear(config["expansion"] * config["channel_size"], config["channel_size"]))
        
        self.norm = nn.LayerNorm(config["channel_size"])
        self.drop = nn.Dropout(config["dropout"])
        
    def forward(self,x):
        residual = x
        x = self.norm(x)
        x, _ = self.multi_head_attention(x, x, x)
        x = self.drop(x)
        x += residual
        
        residual = x
        x = self.norm(x)
        x = self.feed_foward(x)
        x = self.drop(x)
        x += residual
        
        return x
        
        
class TransformerEncoder(nn.Module):

    def __init__(self,config):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderBlock(config) for _ in range(config['num_attention'])])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class EEGEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shallow_net = ShallowEncoderNet(config)
        self.n_layers = config["num_attention"]
        self.trns_enc = TransformerEncoder(config)

    def forward(self, inputs):
        x = self.shallow_net(inputs)
        x = x.permute(0, 2, 3, 1).squeeze(1)
        x = self.trns_enc(x)
        return x
    
    
class ShallowDecoderNet(nn.Module):
    def __init__(self, config):
        super(ShallowDecoderNet, self).__init__()
        self.temporal_conv = nn.ConvTranspose2d(in_channels=config["channel_size"], out_channels=config["in_channels"],
                                       kernel_size=(1, config["temporal_conv_size"]), stride=(1, 1))
        self.spatial_conv = nn.ConvTranspose2d(in_channels=config["channel_size"], out_channels=config["channel_size"],
                                      kernel_size=(config["spatial_conv_size"], 1), stride=(1, 1))
        self.bn = nn.BatchNorm2d(config["channel_size"])
        self.elu = nn.ELU()
        self.upsample = nn.Upsample((1, 104))
        #self.upsample = nn.ConvTranspose2d(~~)
        self.dr = nn.Dropout(config["dropout"])
        self.enhance_conv = nn.Conv2d(config["channel_size"], config["channel_size"], (1, 1), stride=(1, 1)) #could enhance fiting ability slightly
    
    def forward(self, inputs):
        x = self.enhance_conv(inputs)
        #x = self.dr(x)
        x = self.upsample(x)
        x = self.elu(x)
        x = self.bn(x)
        x = self.spatial_conv(x)
        x = self.temporal_conv(x)

        return x
    

class EEGDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.shallow_net = ShallowDecoderNet(config)

    def forward(self, encoded):
        if encoded.dim() !=3 :
            encoded = encoded.view(-1,16,40) #(N,16,40) 
        x = encoded.unsqueeze(1).permute(0,3,1,2)  
        x = self.shallow_net(x)
        
        return x
    
class EEGAE(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = EEGEncoder(config)
        self.decoder = EEGDecoder(config)

    def forward(self, inputs):

        z = self.encoder(inputs) #(N,100) / inputs : (N,1,60,128)
        out = self.decoder(z)
        return out