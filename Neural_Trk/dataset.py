import os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import pandas as pd
import numpy as np


class CNNSimDataset(Dataset):
    def __init__(self, inputs, targets, scaler=None):
        self.inputs = inputs
        self.targets = targets
        self.scaler = scaler

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def preparing_trial_data(config):
    path_list = sorted(glob(os.path.join(config["csv_folder"], "*")), key=lambda x: int(x.split(".")[1].split("_")[-1]))
    inputs_list = []
    targets_list = []
    for path in path_list:
        df = pd.read_csv(path)
        eeg = df.iloc[:, :60]
        attn = df["attn"]
        uattn = df["uattn"]

        eeg_numpy = eeg.to_numpy()
        attn_numpy = attn.to_numpy()
        uattn_numpy = uattn.to_numpy()

        eeg_data = []
        for i in range(config["delay"], eeg.shape[0] - config["temporal_size"], config["temporal_step"]):
            cut_eeg = eeg_numpy[i:i + config["temporal_size"]]
            eeg_data.append(cut_eeg)
        attn_data = []
        for i in range(0, len(attn) - config["temporal_size"] - config["delay"], config["temporal_step"]):
            cut_attn = attn_numpy[i:i + config["temporal_size"]]
            attn_data.append(cut_attn)
        uattn_data = []
        for i in range(0, len(uattn) - config["temporal_size"] - config["delay"], config["temporal_step"]):
            cut_uattn = uattn_numpy[i:i + config["temporal_size"]]
            uattn_data.append(cut_uattn)

        eeg_data = np.array(eeg_data)
        attn_data = np.expand_dims(np.array(attn_data), 2)
        uattn_data = np.expand_dims(np.array(uattn_data), 2)

        true_inputs = np.concatenate([eeg_data, attn_data], axis=2)
        false_inputs = np.concatenate([eeg_data, uattn_data], axis=2)
        true_targets = np.ones(shape=(true_inputs.shape[0], 1))
        false_targets = np.zeros(shape=(false_inputs.shape[0], 1))

        inputs = np.concatenate([true_inputs, false_inputs], axis=0)
        targets = np.concatenate([true_targets, false_targets], axis=0)

        inputs_list.append(np.transpose(inputs, (0, 2, 1)))
        targets_list.append(targets)
    return inputs_list, targets_list


def preparing_sim_data(config):
    print("CNN SIM model data preparing...")
    df = pd.read_csv(config["csv_path"])
    eeg = df.iloc[:, :60]
    attn = df["attn"]
    uattn = df["uattn"]

    eeg_numpy = eeg.to_numpy()
    attn_numpy = attn.to_numpy()
    uattn_numpy = uattn.to_numpy()

    eeg_data = []
    for i in range(config["delay"], eeg.shape[0] - config["temporal_size"], config["temporal_step"]):
        cut_eeg = eeg_numpy[i:i+config["temporal_size"]]
        eeg_data.append(cut_eeg)
    attn_data = []
    for i in range(0, len(attn) - config["temporal_size"] - config["delay"], config["temporal_step"]):
        cut_attn = attn_numpy[i:i + config["temporal_size"]]
        attn_data.append(cut_attn)
    uattn_data = []
    for i in range(0, len(uattn) - config["temporal_size"] - config["delay"], config["temporal_step"]):
        cut_uattn = uattn_numpy[i:i + config["temporal_size"]]
        uattn_data.append(cut_uattn)

    eeg_data = np.array(eeg_data)
    attn_data = np.expand_dims(np.array(attn_data), 2)
    uattn_data = np.expand_dims(np.array(uattn_data), 2)

    true_inputs = np.concatenate([eeg_data, attn_data], axis=2)
    false_inputs = np.concatenate([eeg_data, uattn_data], axis=2)
    true_targets = np.ones(shape=(true_inputs.shape[0], 1))
    false_targets = np.zeros(shape=(false_inputs.shape[0], 1))

    train_size = int(true_inputs.shape[0] * config["train_ratio"])

    train_true_inputs = true_inputs[:train_size]
    train_true_targets = true_targets[:train_size]
    train_false_inputs = false_inputs[:train_size]
    train_false_targets = false_targets[:train_size]

    valid_true_inputs = true_inputs[train_size:]
    valid_true_targets = true_targets[train_size:]
    valid_false_inputs = false_inputs[train_size:]
    valid_false_targets = false_targets[train_size:]

    train_inputs = np.concatenate([train_true_inputs, train_false_inputs], axis=0)
    train_targets = np.concatenate([train_true_targets, train_false_targets], axis=0)
    valid_inputs = np.concatenate([valid_true_inputs, valid_false_inputs], axis=0)
    valid_targets = np.concatenate([valid_true_targets, valid_false_targets], axis=0)
    print("CNN SIM model data preparing end")

    return np.transpose(train_inputs, (0, 2, 1)), train_targets, np.transpose(valid_inputs, (0, 2, 1)), valid_targets


def preparing_exp_data(config):
    
    if isinstance(config["csv_folder"], list):
        print('list')
        path_list = []
        for i in range(len(config['csv_folder'])):
            temp_list = sorted(glob(config['csv_folder'][i]))
            path_list.extend(temp_list)
    else:
        print('not list')
        path_list = sorted(glob(config["csv_folder"]))
        
    if 'exp_2' in config['data_name']:
        import h5py
        filepath = "./data/DATA(3conditions)_rev210330.mat"
        matfile = h5py.File(filepath, 'r')
        subj_dat = matfile['DATA']['INDEX']

        exp_2_dict = {}
        exp=2
        for subj in range(10):
            subj_ds = matfile[subj_dat[exp][0]]
            exp_2_dict[f'subj_{subj}'] = np.array(subj_ds[subj_ds['left_is_1'][subj][0]])[0].astype(int).tolist()
            
        final_path_list = []
        for key in exp_2_dict:
            if 'left' in config['data_name']:
                ids = np.where(np.array(exp_2_dict[key])==1)
            else:
                ids = np.where(np.array(exp_2_dict[key])==0)
                
            temp = [i for i in path_list if key in i.split('/')[-1]]
            temp = [i for i in temp if int(i.split('/')[-1].split('_')[-1][:-4]) in ids[0]]
            print(ids)
            print([i.split('/')[-1] for i in temp])
            final_path_list.extend(temp)
        path_list = final_path_list

    inputs_list = []
    targets_list = []
    for path in path_list:
        df = pd.read_csv(path)
        eeg = df.iloc[:, :60]
        attn = df["attn"]
        uattn = df["uattn"]

        eeg_numpy = eeg.to_numpy()
        attn_numpy = attn.to_numpy()
        uattn_numpy = uattn.to_numpy()

        eeg_data = []
        for i in range(config["delay"], eeg.shape[0] - config["temporal_size"], config["temporal_step"]):
            cut_eeg = eeg_numpy[i:i + config["temporal_size"]]
            eeg_data.append(cut_eeg)
        attn_data = []
        for i in range(0, len(attn) - config["temporal_size"] - config["delay"], config["temporal_step"]):
            cut_attn = attn_numpy[i:i + config["temporal_size"]]
            attn_data.append(cut_attn)
        uattn_data = []
        for i in range(0, len(uattn) - config["temporal_size"] - config["delay"], config["temporal_step"]):
            cut_uattn = uattn_numpy[i:i + config["temporal_size"]]
            uattn_data.append(cut_uattn)

        eeg_data = np.array(eeg_data)
        attn_data = np.expand_dims(np.array(attn_data), 2)
        uattn_data = np.expand_dims(np.array(uattn_data), 2)

        true_inputs = np.concatenate([eeg_data, attn_data], axis=2)
        false_inputs = np.concatenate([eeg_data, uattn_data], axis=2)
        true_targets = np.ones(shape=(true_inputs.shape[0], 1))
        false_targets = np.zeros(shape=(false_inputs.shape[0], 1))

        inputs = np.concatenate([true_inputs, false_inputs], axis=0)
        targets = np.concatenate([true_targets, false_targets], axis=0)

        inputs_list.append(np.transpose(inputs, (0, 2, 1)))
        targets_list.append(targets)
    return inputs_list, targets_list


def get_sim_dataset(config):
    train_inputs, train_targets, valid_inputs, valid_targets = preparing_sim_data(config)
    train_dataset = CNNSimDataset(train_inputs, train_targets)
    valid_dataset = CNNSimDataset(valid_inputs, valid_targets)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"])

    return train_loader, valid_loader, valid_inputs, valid_targets
