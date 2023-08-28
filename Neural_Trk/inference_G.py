import torch
import numpy as np
import pandas as pd
import random
import yaml
import os
from glob import glob
import argparse
from models import models
import pickle
import warnings
import json


warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description='LOOCV inference_G file')
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(config["seed"])

    if isinstance(config['csv_folder'],list):
        path_list1 = sorted(glob(config['csv_folder'][0]))
        path_list2 = sorted(glob(config['csv_folder'][1]))
        path_list = path_list1 + path_list2
    else:
        path_list = sorted(glob(config["csv_folder"]))

    if args.scaler != None:
        result_dir = os.path.join("./results_G/prediction/", config["data_name"] + "_scaled_64step")
    else:
        result_dir = os.path.join("./results_G/prediction/", config["data_name"] + "_64step")
    os.makedirs(result_dir, exist_ok=True)

    for t, idx in enumerate(range(0,len(path_list),30)):
        true_pred_list = []
        false_pred_list = []
        
        true_input_list = []
        false_input_list = []
        
        if args.scaler != None:
            scaler_save_folder = os.path.join("./results_G/scaler", config["data_name"])
            scaler_save_path = os.path.join(scaler_save_folder, "Subj{}.pickle".format(t))
            
            with open(scaler_save_path, 'rb') as f:
                scaler = pickle.load(f)
        
        for path in path_list[idx:idx+30]: #subj
            df = pd.read_csv(path)
            eeg = df.iloc[:, :60]
            attn = df["attn"]
            uattn = df["uattn"]

            eeg_numpy = eeg.to_numpy()
            attn_numpy = attn.to_numpy()
            uattn_numpy = uattn.to_numpy()

            eeg_data = []
            for i in range(0, eeg.shape[0] - config["temporal_size"] + 1, config["temporal_size"]):
                cut_eeg = eeg_numpy[i:i + config["temporal_size"]]
                eeg_data.append(cut_eeg)
            attn_data = []
            for i in range(0, len(attn) - config["temporal_size"] + 1, config["temporal_size"]):
                cut_attn = attn_numpy[i:i + config["temporal_size"]]
                attn_data.append(cut_attn)
            uattn_data = []
            for i in range(0, len(uattn) - config["temporal_size"] + 1, config["temporal_size"]):
                cut_uattn = uattn_numpy[i:i + config["temporal_size"]]
                uattn_data.append(cut_uattn)
                

            eeg_data = np.array(eeg_data) #batch,128,60
            attn_data = np.expand_dims(np.array(attn_data), 2)
            uattn_data = np.expand_dims(np.array(uattn_data), 2)
            
            if args.scaler != None:
                eeg_data = scaler.transform(eeg_data.reshape(-1, 60))
                eeg_data = eeg_data.reshape(-1,config["temporal_size"],60)

            true_inputs = np.concatenate([eeg_data, attn_data], axis=2)
            false_inputs = np.concatenate([eeg_data, uattn_data], axis=2)
            
            scaled_true_input = true_inputs.transpose((0, 2, 1))
            scaled_false_input = false_inputs.transpose((0, 2, 1))
            
            true_input_list.append(scaled_true_input)
            false_input_list.append(scaled_false_input)
            
        true_inputs_concat = np.concatenate(true_input_list, axis=0) 
        false_inputs_concat = np.concatenate(false_input_list, axis=0)
        # print(true_inputs_concat.shape) = 900,61,128

        model = models.CNNSim()
        if args.scaler != None:
            model_path = os.path.join("./results_G/scaled_saved_model", config["data_name"], "Subj{}.pt".format(t))
        else:
            model_path = os.path.join("./results_G/saved_model", config["data_name"], "Subj{}.pt".format(t))
        print("Subj{}.pt model loaded".format(t))
        model.load_state_dict(torch.load(model_path))
        model.eval()
        model = model.to(device)
        
        for c in range(true_inputs_concat.shape[0]):
            cut_true_inputs = true_inputs_concat[np.newaxis, c, ...]
            cut_false_inputs = false_inputs_concat[np.newaxis, c, ...]
            cut_true_inputs = torch.from_numpy(cut_true_inputs).to(device).float()
            cut_false_inputs = torch.from_numpy(cut_false_inputs).to(device).float()
            true_pred = model(cut_true_inputs)
            false_pred = model(cut_false_inputs)

            true_pred_list.append(true_pred.cpu().detach().numpy()[0][0])
            false_pred_list.append(false_pred.cpu().detach().numpy()[0][0])

        result_path = os.path.join(result_dir, "Subj{}.csv".format(t))
        result = pd.DataFrame({"True_pred": true_pred_list,
                            "False_pred": false_pred_list})
        result.to_csv(result_path, index=False)
