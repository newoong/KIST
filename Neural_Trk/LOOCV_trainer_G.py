import os
import random
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dataset import dataset
from models import models
import numpy as np
import pandas as pd
import pickle


parser = argparse.ArgumentParser(description='LOOCV train file')
parser.add_argument('--config_file', type=str, required=True, help="config file path")
parser.add_argument('--scaler', type=str, default=None, help='if need scaling, write scaler name to use(MinMax, Standard)')

args = parser.parse_args()


def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)


if __name__ == "__main__":
    with open(args.config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    seed_everything(config["seed"])

    inputs, targets = dataset.preparing_exp_data(config)
    
    print('inputs set done')
    
    if args.scaler==None:
        model_save_folder = os.path.join("./results_G/saved_model", config["data_name"])
    else:
        model_save_folder = os.path.join("./results_G/scaled_saved_model", config["data_name"])
    print(f'model will be saved at {model_save_folder}')
        
    os.makedirs(model_save_folder, exist_ok=True)

    def train_step(model, loader, optimizer, criterion, epoch, device):
        model.train()
        epoch_loss = 0

        for batch, (x, y) in enumerate(loader):
            x = x.to(device).float()
            y = y.to(device).float()

            optimizer.zero_grad()

            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if batch % 20 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch * len(x), len(loader.dataset),
                           100. * batch / len(loader), loss.item()))

        return epoch_loss / len(loader)


    def evaluate_step(model, loader, criterion, device):
        model.eval()
        epoch_loss = 0
        preds = []
        for batch, (x, y) in enumerate(loader):
            x = x.to(device).float()
            y = y.to(device).float()

            outputs = model(x)
            loss = criterion(outputs, y)
            epoch_loss += loss.item()
            preds.extend(outputs.cpu().detach().numpy())
        
        true_preds = []
        false_preds = []
        bs = loader.batch_size
        for idx in range(0,len(preds),bs):
            true_preds.extend(preds[idx:idx+(bs//2)])
            false_preds.extend(preds[idx+(bs//2):idx+bs])
            
        true_preds = np.array(true_preds)
        false_preds = np.array(false_preds)
        count = np.sum(true_preds > false_preds)
        return count / len(true_preds), epoch_loss / len(loader)
    

    print(f'total data size : {len(inputs)},{inputs[0].shape}')
    
    # exp_2로 모델링할 땐 subj마다 left, right 각각 15개씩이니까
    # exp_0, exp_1로 모델링할 때는 subj마다 left, right 각각 30개씩이니까
    if 'exp_2' in config['data_name']:
        per=15
    else:
        per = 30
    for subj,i in enumerate(range(0,len(inputs),per)):
        print("subj {} validation".format(subj))

        model = models.CNNSim()
        model.to(device)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

        train_inputs = inputs.copy()
        train_targets = targets.copy()

        valid_inputs = inputs[i:i+per]
        valid_targets = targets[i:i+per]
        
        valid_batch = valid_inputs[0].shape[0]
        print('vaild batch_size : {}'.format(valid_batch))

        del train_inputs[i:i+per]
        del train_targets[i:i+per]

        if len(train_inputs) != len(inputs) - per or len(train_targets) != len(inputs) - per:
            raise Exception("del error")

        train_inputs = np.concatenate(train_inputs, axis=0)
        train_targets = np.concatenate(train_targets, axis=0)
        
        valid_inputs = np.concatenate(valid_inputs, axis=0)
        valid_targets = np.concatenate(valid_targets, axis=0)
        
        if args.scaler != None:
        
            if args.scaler == 'MinMax':
                scaler = MinMaxScaler()
            elif args.scaler == 'Standard':
                scaler = StandardScaler()
            else:
                print('No scaler "{}"'.format(args.scaler))
                raise Exception("scaler nono")

            envelope = train_inputs[:,-1,:]
            envelope = np.expand_dims(envelope, 1)
            train_inputs = scaler.fit_transform(np.transpose(train_inputs[:,:60,:], (1, 0, 2)).reshape(60, -1).T).T
            train_inputs = np.transpose(train_inputs.reshape(60, -1, config["temporal_size"]), (1, 0, 2))
            train_inputs = np.concatenate((train_inputs,envelope),axis=1)
              
            envelope = valid_inputs[:,-1,:]
            envelope = np.expand_dims(envelope, 1)
            valid_inputs = scaler.transform(np.transpose(valid_inputs[:,:60,:], (1, 0, 2)).reshape(60, -1).T).T
            valid_inputs = np.transpose(valid_inputs.reshape(60, -1, config["temporal_size"]), (1, 0, 2))
            valid_inputs = np.concatenate((valid_inputs,envelope),axis=1)

            scaler_save_folder = os.path.join("./results_G/scaler", config["data_name"])
            os.makedirs(scaler_save_folder, exist_ok=True)
            scaler_save_path = os.path.join(scaler_save_folder, "Subj{}.pickle".format(subj))
            with open(scaler_save_path, 'wb') as f:
                pickle.dump(scaler, f)

        train_dataset = dataset.CNNSimDataset(train_inputs, train_targets)
        valid_dataset = dataset.CNNSimDataset(valid_inputs, valid_targets)

        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=7)
        valid_loader = DataLoader(valid_dataset, batch_size=valid_batch, shuffle=False, num_workers=7)
        
        print('data prepare done !')
        

        train_loss_list = []
        valid_loss_list = []
        valid_acc_list = []
        max_acc = 0
        patient = 0
        
        for epoch in range(1, config["num_epoch"] + 1):
            print("-" * 50)
            train_loss = train_step(model, train_loader, optimizer, criterion, epoch, device)
            valid_acc, valid_loss = evaluate_step(model, valid_loader, criterion, device)

            train_loss_list.append(train_loss)
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)

            print(f"\nEPOCH {epoch:>3d} Train Loss: {train_loss:>.5f} Validation Loss: {valid_loss:>.5f} Validation Accuracy: {valid_acc:>.5f}\n")

            os.makedirs("./temp_G", exist_ok=True)
            if valid_acc > max_acc:
                max_acc = valid_acc
                torch.save(model.state_dict(), "./temp_G/temp.pt")
                print(f"save at epoch{epoch:>3d}")
                patient = 0
            else:
                if epoch >= 150:
                    patient += 1
            if patient > 50:
                print(f"EPOCH{epoch:>3d} stop")
                model.load_state_dict(torch.load("./temp_G/temp.pt"))
                break

        print("-" * 50)
        print("-" * 50)
        print()

        del train_inputs
        del train_targets


        model_save_path = os.path.join(model_save_folder, "Subj{}.pt".format(subj))
        torch.save(model.state_dict(), model_save_path)
        if args.scaler == None:
            loss_save_folder = os.path.join("./results_G/loss", config["data_name"])
        else:
            loss_save_folder = os.path.join("./results_G/scaled_loss", config["data_name"])
            
        os.makedirs(loss_save_folder, exist_ok=True)

        loss_path = os.path.join(loss_save_folder, "Subj{}.csv".format(subj))
        loss_df = pd.DataFrame({"train_loss": train_loss_list,
                                "valid_loss": valid_loss_list,
                                "valid_acc": valid_acc_list})
        loss_df.to_csv(loss_path, index=False)
