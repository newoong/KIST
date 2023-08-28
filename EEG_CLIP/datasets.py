from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os
import pandas as pd


class AudioDataset(Dataset):
    def __init__(self, wav_list, sr, time_duration, time_stride,
                 window_size, window_stride, n_mels, slicing, prepared):
        super().__init__()
        self.wav_list = wav_list
        self.sr = sr
        self.time_duration = time_duration
        self.time_stride = time_stride
        self.window_size = window_size
        self.window_stride = window_stride
        self.n_mels = n_mels
        self.slicing = slicing
        self.prepared=prepared

        self.wav_seq_len = sr * 60  # 60 seconds files
        self.hop_size = int(round(window_stride * sr))
        self.n_fft = int(round(window_size * sr))

        self.sample_len = int(time_duration / window_stride)
        self.wav_per_samples = (self.wav_seq_len // self.hop_size + 1) - self.sample_len

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        if self.prepared:
            s = pd.read_csv(self.wav_list[idx])
            s = s.values[np.newaxis, :, :]
        else:
            y, sr = librosa.load(self.wav_list[idx], sr=self.sr)
            s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, n_mels=self.n_mels,
                                            hop_length=self.hop_size)
            s_max = np.max(s)
            s_min = np.min(s)
            s = (s - s_min) / (s_max - s_min)   # min max scaling
            s = s[np.newaxis, ..., :self.sample_len]
        return s

    def get_input_shape(self):
        return [1, self.slicing, self.sample_len]


def get_audio_loader(wav_list, sr, time_duration, time_stride, window_size, window_stride, n_mels, slicing, prepared, batch_size, train):
    dataset = AudioDataset(wav_list, sr, time_duration, time_stride, window_size, window_stride, n_mels, slicing, prepared)

    if train:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=6, shuffle=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=6, shuffle=False)

    return loader


class EEGDataset(Dataset):
    def __init__(self, eeg_list):
        super().__init__()
        self.eeg_list = eeg_list

    def __len__(self):
        return len(self.eeg_list)

    def __getitem__(self, idx):
        eeg = pd.read_csv(self.eeg_list[idx]).to_numpy().T[np.newaxis, ...]

        return eeg


def get_eeg_loader(eeg_list, batch_size, train):
    dataset = EEGDataset(eeg_list)

    if train:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=6, shuffle=True)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=6, shuffle=False)

    return loader


class CLIPDataset(Dataset):
    def __init__(self, csv_dir, eeg_list, audio_config):
        self.eeg_list = eeg_list
        self.csv_dir = csv_dir
        # self.sr = audio_config["sr"]
        # self.n_fft = int(round(audio_config["window_size"] * audio_config["sr"]))
        # self.n_mels = audio_config["n_mels"]
        self.slicing = audio_config['slicing']
        # self.hop_size = int(round(audio_config["window_stride"] * audio_config["sr"]))
        self.sample_len = int(audio_config["time_duration"] / audio_config["window_stride"])

    def __len__(self):
        return len(self.eeg_list)

    def __getitem__(self, idx):
        eeg_path = self.eeg_list[idx]
        csv_path = self._get_csv_path(self.csv_dir, eeg_path)

        eeg = pd.read_csv(eeg_path).to_numpy().T[np.newaxis, ...]
        s = pd.read_csv(csv_path).to_numpy()[np.newaxis, ...]
        # y, sr = librosa.load(wav_path, sr=self.sr)
        # s = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, n_mels=self.n_mels,
        #                                    hop_length=self.hop_size)
        # s_max = np.max(s)
        # s_min = np.min(s)
        # s = (s - s_min) / (s_max - s_min)  # min max scaling
        # s = s[np.newaxis, ..., :self.sample_len]

        return eeg, s

    def _get_csv_path(self, csv_dir, eeg_path):
        split = eeg_path.split("/")

        exp_cond = split[3]
        name = split[-1]

        name_split = name.replace(".csv", "").split("_")

        if int(exp_cond[-1]) == 0:
            wav_path = os.path.join(self.csv_dir, exp_cond, name_split[1] + "_" + name_split[-1] + ".csv")
        elif int(exp_cond[-1]) == 1:
            if name_split[1][0] == "L":
                wav_path = os.path.join(self.csv_dir, exp_cond, "Yulgok" + name_split[1][1:] +
                                        "_" + name_split[-1] + ".csv")
            else:
                wav_path = os.path.join(self.csv_dir, exp_cond, "Kant" + name_split[1][1:] +
                                        "_" + name_split[-1] + ".csv")
        else:
            wav_path = os.path.join(self.csv_dir, exp_cond, name_split[1] + "_" + name_split[-1] + ".csv")
        return wav_path

    def get_audio_input_shape(self):
        return [1, self.slicing, self.sample_len]


def get_clip_loader(csv_dir, eeg_list, audio_config, batch_size):
    dataset = CLIPDataset(csv_dir, eeg_list, audio_config)

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=6, shuffle=True)
    return loader
