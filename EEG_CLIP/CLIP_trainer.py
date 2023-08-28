import os
import torch
import torch.nn.functional as F
from torch import nn
from models import EEGConformer, EEGEncoder, AudioEncoder, Projection
from utils import seed_everything, load_config
from dataset import get_clip_loader
from glob import glob
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Training CLIP')
parser.add_argument('--audio_encoder_path', type=str,
                    default="./ckpt_temp/saved_model/audio_encoder/2023-06-13-16:27_audio_encoder.pt",
                    help='audio encoder path')
parser.add_argument('--eeg_encoder_path', type=str,
                    default="./ckpt_temp/saved_model/eeg_encoder/2023-06-13-16:27_eeg_encoder.pt",
                    help='eeg encoder path')
parser.add_argument('--no_projection', action = 'store_true', help='encoder projection head')
parser.add_argument('--freeze', action = 'store_true', help='freeze backbone')

args = parser.parse_args()

class add_proj_enc(nn.Module):
    def __init__(self, encoder, proj_enc):
        super(add_proj_enc,self).__init__()
        self.encoder = encoder
        self.proj_enc = proj_enc

    def forward(self,inputs):
        x = self.encoder(inputs)
        x = self.proj_enc(x)
        
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('device :',device)

    config = load_config("./configs/CLIP_trainer.yaml")
    eeg_data_config = config["eeg_data"]
    eeg_model_config = config["EEGNet"]
    train_config = config["train"]
    audio_data_config = config["audio_data"]
    audio_model_config = config["AudioNet"]

    seed_everything(train_config["seed"])
    
    if args.no_projection:
        save_dir = './ckpt_temp/noproj_clip'
    else:
        save_dir = './ckpt_temp/proj_clip'
    os.makedirs(save_dir, exist_ok=True)
        
    csv_dir = "./data/split_wav_csv"
    eeg_list = sorted(glob("./data/split_eeg/*/*.csv"))
    # eeg_list = sorted(glob("./data/split_eeg/exp_0/subj0_*.csv")) ## exp_anything? exp_specifiy?
    clip_loader = get_clip_loader(csv_dir, eeg_list, audio_data_config, train_config["batch_size"])

    audio_input_shape = clip_loader.dataset.get_audio_input_shape()
    
    audio_encoder = AudioEncoder(audio_input_shape, audio_model_config["n_layer"], audio_model_config["channels"])
    eeg_encoder = EEGEncoder(eeg_model_config)
    
    dim = (audio_input_shape[1] // (2**audio_encoder.n_layer)) * (audio_input_shape[2] // (2**audio_encoder.n_layer)) * audio_encoder.channels[-1]
    
    ##-----------------append projection-------------------------------#
    if args.no_projection:        
        audio_encoder.load_state_dict(torch.load(args.audio_encoder_path))
        audio_encoder = add_proj_enc(audio_encoder,Projection(dim, audio_model_config['hidden_dim'], audio_model_config['latent_dim']))
        print('audio_encoder loaded complete')
        eeg_encoder.load_state_dict(torch.load(args.eeg_encoder_path))
        eeg_encoder = add_proj_enc(eeg_encoder,Projection(16*40, eeg_model_config['hidden_dim'], eeg_model_config['latent_dim']))
        print('eeg_encoder loaded complete')
    else:
        audio_encoder = add_proj_enc(audio_encoder,Projection(dim, audio_model_config['hidden_dim'], audio_model_config['latent_dim']))
        eeg_encoder = add_proj_enc(eeg_encoder,Projection(16*40, eeg_model_config['hidden_dim'], eeg_model_config['latent_dim']))
        audio_encoder.load_state_dict(torch.load(args.audio_encoder_path))
        print('audio_encoder loaded complete')
        eeg_encoder.load_state_dict(torch.load(args.eeg_encoder_path))
        print('eeg_encoder loaded complete')
    ##-----------------append projection-------------------------------#
    
    if args.freeze:
        for param in audio_encoder.encoder.parameters():
            param.required_grad = False
            
        for param in eeg_encoder.encoder.parameters():
            param.required_grad = False

    audio_encoder = audio_encoder.to(device)
    eeg_encoder = eeg_encoder.to(device)

    optimizer = torch.optim.Adam(params=list(audio_encoder.parameters()) + list(eeg_encoder.parameters()),
                                 lr=train_config["lr"])
    loss_list = []
    audio_encoder.train()
    eeg_encoder.train()
    for epoch in range(train_config["num_epoch"] + 1):
        epoch_loss = 0
        for eeg, mel in clip_loader:
            mel = mel.float().to(device)
            eeg = eeg.float().to(device)

            audio_vector = audio_encoder(mel) #(N,100)
            eeg_vector = eeg_encoder(eeg)     #(N,100)

            norm_audio = F.normalize(audio_vector, p=2., dim=1)
            norm_eeg = F.normalize(eeg_vector, p=2., dim=1)

            logits = torch.matmul(norm_audio, norm_eeg.transpose(1, 0)) #(N,N) : 100 size vector dot product

            labels = torch.arange(logits.shape[0], device=device) 

            loss_audio = F.cross_entropy(logits, labels)
            loss_eeg = F.cross_entropy(logits.transpose(1, 0), labels)
            loss = (loss_audio + loss_eeg) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        loss_list.append(epoch_loss / len(clip_loader))
        print(f"EPOCH {epoch:>3d} loss: {epoch_loss / len(clip_loader):>6f} ")
        
        if epoch==0:
            ckpt=epoch_loss
        else:
            if ckpt > epoch_loss:   
                ckpt = epoch_loss   
                torch.save(audio_encoder.state_dict(), os.path.join(save_dir, "audio.ckpt"))
                torch.save(eeg_encoder.state_dict(), os.path.join(save_dir, "eeg.ckpt"))

    plt.figure(figsize=(12, 6))
    plt.plot(loss_list, label="loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "CLIP_loss.png"))
    plt.title("Loss", size=12)
