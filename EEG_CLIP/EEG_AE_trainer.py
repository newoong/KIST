import os
import torch
import torch.nn as nn
from dataset import get_eeg_loader
from utils import train_test_split, load_config, seed_everything
from models import EEGAE, EEGEncoder, Projection
from glob import glob
import wandb
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Training EEG auto encoder model')
#parser.add_argument('--name', type=str, required=True, help='model name(for wandb)')
parser.add_argument('--add_projection', action='store_true',help='add projection layer to encoder')
#parser.add_argument('--data_path', type=str, default='./data/split_eeg/*/*.csv', required=True, help='eeg_data path')


args = parser.parse_args()

class CustomLoss1(nn.Module): ##flatten EEG data
    def __init__(self):
        super(CustomLoss1, self).__init__()

    def forward(self, x, reconstructed_x):
        x = x.reshape(x.shape[0],-1)
        reconstructed_x = reconstructed_x.reshape(reconstructed_x.shape[0],-1)
        x_norm = F.normalize(x, dim=-1)
        reconstructed_x_norm = F.normalize(reconstructed_x, dim=-1)
        dot = torch.mean(torch.sum(x_norm * reconstructed_x_norm,dim=-1))
        loss = 1 - dot  # loss 계산
        
        return loss
    
class CustomLoss2(nn.Module): ##per channel
    def __init__(self):
        super(CustomLoss2, self).__init__()

    def forward(self, x, reconstructed_x):
        x_norm = F.normalize(x, dim=-1)  # x의 norm 계산
        reconstructed_x_norm = F.normalize(reconstructed_x, dim=-1)  # 재구성된 x의 norm 계산
        dot = torch.sum(x_norm * reconstructed_x_norm, dim=-1)
        loss = torch.mean(loss)
        loss = 1 - dot  # loss 계산
        
        return loss
    
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



def train(model, loader, loss_fn, optim, device, input_drop):
    model.train()
    train_loss = 0
    for i, x in enumerate(loader):
        x = x.float().to(device)

        if input_drop:
            x = torch.dropout(x, input_drop,train=True)
        out = model(x)

        loss = loss_fn(out, x)
        optim.zero_grad()
        loss.backward()
        optim.step()

        train_loss += loss.item()
    return train_loss / len(loader)


def evaluate(model, loader, loss_fn, device):
    model.eval()

    eval_loss = 0
    with torch.no_grad():
        for x in loader:
            x = x.float().to(device)

            out = model(x)
            loss = loss_fn(out, x)

            eval_loss += loss.item()
    return eval_loss / len(loader)


if __name__ == "__main__":
    #name = args.name
    if args.add_projection:
        save_dir = "./ckpt_temp/EEG_AE_proj"
    else:
        save_dir = "./ckpt_temp/EEG_AE_noproj"
    
    os.makedirs(save_dir, exist_ok=True)

    data_path = './data/split_eeg/*/*.csv'
    file_list = glob(data_path)  
        
    config = load_config("./configs/EEG_trainer.yaml")

    eeg_configs = config["EEG"]
    data_configs = config["data"]
    train_configs = config["train"]

    seed_everything(train_configs["seed"])

    '''wandb.init(project="EEG_CLIP",
               config=config)
    wandb.run.name = name + "_" + str(datetime.now().strftime('%Y-%m-%d-%H:%M'))'''

    # data set preparing
    #can use split_wav_csv directly(already transformed to mel_csv)
    train_file, test_file = train_test_split(file_list, train_configs["train_ratio"])
 
    train_loader = get_eeg_loader(train_file, train_configs["batch_size"], True)
    test_loader = get_eeg_loader(test_file, train_configs["batch_size"], False)

    # model load
    EEG_AE = EEGAE(eeg_configs)
    
    if args.add_projection:
        proj_enc = Projection(16*40, eeg_configs['hidden_dim'], eeg_configs['latent_dim'])
        proj_dec = Projection(eeg_configs['latent_dim'],eeg_configs['hidden_dim'],16*40)
        EEG_AE = add_proj_ae(EEG_AE.encoder, proj_enc, proj_dec, EEG_AE.decoder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #criterion = nn.BCELoss() ##predict value must 0~1 -> cant do with EEG
    #criterion = nn.L1Loss()
    #criterion = nn.MSELoss()
    criterion = CustomLoss1()
    #criterion = CustomLoss2()
    
    optimizer = torch.optim.Adam(EEG_AE.parameters(), lr=float(train_configs["lr"]))
    EEG_AE = EEG_AE.to(device)

    train_losses=[]
    valid_losses=[]
    for epoch in range(1, train_configs["num_epoch"] + 1):
        train_loss = train(EEG_AE, train_loader, criterion, optimizer, device, eeg_configs["input_drop"])
        valid_loss = evaluate(EEG_AE, test_loader, criterion, device)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f"EPOCH {epoch:>3d} loss: train_loss - {train_loss:>6f} / valid_loss - {valid_loss:>6f} ")

        if epoch%10==0:
            save_path = os.path.join(save_dir, "epoch_{}.pt".format(epoch))
            torch.save(EEG_AE.state_dict(), save_path)

    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="train_loss")
    plt.plot(valid_losses, label="valid_loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir,"EEG_AE_loss.png"))
    plt.title("Loss", size=12)


        # wandb.log({
        #     "train_loss": train_loss,
        #     "validation_loss": valid_loss
        # })
