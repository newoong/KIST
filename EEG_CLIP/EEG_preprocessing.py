import numpy as np
import h5py
import os
import pandas as pd
from tqdm import tqdm
from utils import load_config


def wav_path_check(exp_cond_idx, trial_idx, sample_num, is_left=None, a_idx=None):
    if exp_cond_idx == 0:
        if is_left == 0:
            wav_path = "./data/split_wav/exp_0/R{}_sample{}.wav".format(trial_idx + 1, sample_num)
        else:
            wav_path = "./data/split_wav/exp_0/L{}_sample{}.wav".format(trial_idx + 1, sample_num)
    elif exp_cond_idx == 1:
        if is_left == 0:
            wav_path = "./data/split_wav/exp_1/Kant{}_sample{}.wav".format(trial_idx + 1, sample_num)
        else:
            wav_path = "./data/split_wav/exp_1/Yulgok{}_sample{}.wav".format(trial_idx + 1, sample_num)
    else:
        wav_path = "./data/split_wav/exp_2/s{}_sample{}.wav".format(a_idx, sample_num)
    return os.path.exists(wav_path)


# config setting
config = load_config("./configs/EEG_trainer.yaml")["data"]

save_dir_root = "./data/split_eeg"
filepath = "./data/DATA(3conditions)_rev210330.mat"
matfile = h5py.File(filepath, 'r')

eeg_dat = matfile['DATA']['EEG']
subj_dat = matfile['DATA']['INDEX']
speech_env = matfile['DATA']['SPEECH']
exp_cond = matfile['DATA']['Label']


N_exp_con = 3   # Number of experiment condition
N_subj = 10     # Number of subject in each condition

for exp_cond_inx in tqdm(range(N_exp_con)):
    save_dir = os.path.join(save_dir_root, "exp_{}".format(exp_cond_inx))
    os.makedirs(save_dir, exist_ok=True)
    for subj_inx in range(N_subj):
        eeg_ds = matfile[eeg_dat[exp_cond_inx][0]]
        raw_eeg = np.array(eeg_ds[eeg_ds['data'][subj_inx][0]])

        [eegL, nCh, nTrial] = raw_eeg.shape

        subj_ds = matfile[subj_dat[exp_cond_inx][0]]
        if exp_cond_inx < 2:
            is_left_list = matfile[subj_ds["left_is_1"][subj_inx][0]][0]
            wav_file = "./data/split_wav/exp_{}/".format(exp_cond_inx)
            for trial_inx, is_left in enumerate(is_left_list):
                for i, cut_inx in enumerate(range(0, eegL - config["sr"] * config["time_duration"], config["sr"])):
                    #####
                    if not wav_path_check(exp_cond_inx, trial_inx, i, is_left=is_left):
                        continue
                    #####
                    cut_eeg = raw_eeg[cut_inx:cut_inx + (config["sr"] * config["time_duration"]), :, trial_inx]
                    if is_left == 0:
                        file_name = "subj{}_R{}_sample{}.csv".format(subj_inx, trial_inx + 1, i)
                    else:
                        file_name = "subj{}_L{}_sample{}.csv".format(subj_inx, trial_inx + 1, i)
                    save_path = os.path.join(save_dir, file_name)
                    columns = ["ch{}".format(i) for i in range(60)]
                    df = pd.DataFrame(cut_eeg, columns=columns)
                    df.to_csv(save_path, index=False)
        else:
            a_index = matfile[subj_ds['a'][subj_inx][0]][0]
            for trial_inx, a in enumerate(a_index):
                a = int(a)
                for i, cut_inx in enumerate(range(0, eegL - config["sr"] * config["time_duration"], config["sr"])):
                    #####
                    if not wav_path_check(exp_cond_inx, trial_inx, i, a_idx=a):
                        continue
                    #####
                    cut_eeg = raw_eeg[cut_inx:cut_inx + (config["sr"] * config["time_duration"]), :, trial_inx]
                    file_name = "subj{}_s{}_{}_sample{}.csv".format(subj_inx, a, trial_inx + 1, i)
                    save_path = os.path.join(save_dir, file_name)
                    columns = ["ch{}".format(i) for i in range(60)]
                    df = pd.DataFrame(cut_eeg, columns=columns)
                    df.to_csv(save_path, index=False)
