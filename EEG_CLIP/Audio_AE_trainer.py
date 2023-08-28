import os
import torch
import torch.nn as nn
from dataset import get_audio_loader
from utils import train_test_split, load_config, seed_everything
from models import AudioAE, AudioAE1D, Projection
from glob import glob
import wandb
from datetime import datetime
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Training Audio auto encoder model')
#parser.add_argument('--name', type=str, required=True, help='model name(for wandb)')
parser.add_argument('--type', type=str, required=True, help='model type')
parser.add_argument('--prepared', action = 'store_true', required=True, help='use prepared csv or untransformed wav')
parser.add_argument('--add_projection', action='store_true',help='add projection layer to encoder')


args = parser.parse_args()

class add_proj_ae(nn.Module):
    def __init__(self, encoder, proj_enc, proj_dec, decoder):
        super(add_proj_ae,self).__init__()
        self.encoder = encoder
        self.proj_enc = proj_enc
        self.proj_dec = proj_dec
        self.decoder = decoder
    def forward(self,inputs):
        x = self.encoder(inputs)
        x = self.proj_enc(x)
        x = self.proj_dec(x)
        x = self.decoder(x)
        
        return x


def train(model, loader, loss_fn, optim, device, model_type, input_drop):
    model.train()

    train_loss = 0
    for i, x in enumerate(loader):
        x = x.float().to(device)
        if model_type == "1d":
            x = x.squeeze(1)

        if input_drop:
            x = torch.dropout(x, input_drop,train=True)
        out = model(x)

        loss = loss_fn(out, x)
        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss += loss.item()
    return train_loss / len(loader)


def evaluate(model, loader, loss_fn, device, model_type):
    model.eval()

    eval_loss = 0
    with torch.no_grad():
        for x in loader:
            x = x.float().to(device)

            if model_type == "1d":
                x = x.squeeze(1)

            out = model(x)
            loss = loss_fn(out, x)

            eval_loss += loss.item()
    return eval_loss / len(loader)


if __name__ == "__main__":
    #name = args.name
    model_type = args.type
    if args.add_projection:
        save_dir = "./ckpt_temp/Audio_AE_proj"
    else:
        save_dir = "./ckpt_temp/Audio_AE_noproj"
    os.makedirs(save_dir, exist_ok=True)

    if args.prepared:
        print('use prepared csvs!')
        file_list = glob("./data/split_wav_csv/*/*.csv")
    else:
        file_list = glob("./data/split_wav/*/*.wav")  # wav file path
        
    config = load_config("./configs/AE_trainer.yaml")

    model_configs = config["Model"]["AudioNet"]
    data_configs = config["Data"]
    train_configs = config["Train"]

    seed_everything(train_configs["seed"])

    '''wandb.init(project="EEG_CLIP",
               config=config)
    wandb.run.name = name + "_" + str(datetime.now().strftime('%Y-%m-%d-%H:%M'))'''

    # data set preparing
    #can use split_wav_csv directly(already transformed to mel_csv)
    train_file, test_file = train_test_split(file_list, train_configs["train_ratio"])
    train_loader = get_audio_loader(train_file, data_configs["sr"], data_configs["time_duration"],
                                    data_configs["time_stride"], data_configs["window_size"],
                                    data_configs["window_stride"], data_configs["n_mels"], data_configs['slicing'], args.prepared, train_configs["batch_size"],
                                    True)
    test_loader = get_audio_loader(test_file, data_configs["sr"], data_configs["time_duration"],
                                   data_configs["time_stride"], data_configs["window_size"],
                                   data_configs["window_stride"], data_configs["n_mels"], data_configs['slicing'], args.prepared, train_configs["batch_size"],
                                   False)

    # get input shape
    input_shape = train_loader.dataset.get_input_shape()

    # model load
    # if model_type == "1d":
    #     audio_AE = AudioAE1D(input_shape[-1], input_shape[1], model_configs["n_layer"], model_configs["channels"],
    #                          model_configs["latent_dim"])
    # else:
    #     audio_AE = AudioAE(input_shape, model_configs["n_layer"], model_configs["channels"],
    #                        model_configs["latent_dim"])
        
    if model_type == "1d":
        audio_AE = AudioAE1D(input_shape[-1], input_shape[1], model_configs["n_layer"], model_configs["channels"],
                             model_configs["latent_dim"])
    else:
        audio_AE = AudioAE(input_shape, model_configs["n_layer"], model_configs["channels"])
        
    if args.add_projection:
        #dim = (input_shape[1] // (2**audio_AE.n_layer)) * (input_shape[2] // (2**audio_AE.n_layer)) * audio_AE.channels[-1]
        dim = 7500
        proj_enc = Projection(dim, model_configs['hidden_dim'], model_configs['latent_dim'])
        proj_dec = Projection(model_configs['latent_dim'], model_configs['hidden_dim'], dim)
        audio_AE = add_proj_ae(audio_AE.encoder, proj_enc, proj_dec, audio_AE.decoder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.BCELoss() ##why BCE? ##predict value must 0~1
    #criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(audio_AE.parameters(), lr=float(train_configs["lr"]))
    audio_AE = audio_AE.to(device)

    train_losses=[]
    valid_losses=[]
    for epoch in range(1, train_configs["epochs"] + 1):
        train_loss = train(audio_AE, train_loader, criterion, optimizer, device, model_type,
                           model_configs["input_drop"])
        valid_loss = evaluate(audio_AE, test_loader, criterion, device, model_type)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"EPOCH {epoch:>3d} loss: train_loss - {train_loss:>6f} / valid_loss - {valid_loss:>6f} ")

        if epoch%10==0:
            save_path = os.path.join(save_dir, "epoch_{}.pt".format(epoch))
            torch.save(audio_AE.state_dict(), save_path)

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="train_loss")
    plt.plot(valid_losses, label="valid_loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir,"Audio_AE_loss.png"))
    plt.title("Loss", size=12)


        # wandb.log({
        #     "train_loss": train_loss,
        #     "validation_loss": valid_loss
        # })
