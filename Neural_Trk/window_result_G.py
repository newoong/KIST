from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


dir_path = "./results_G/prediction/model_upgrade_exp_2_left_final_64step/*.csv"
subj_list = sorted(glob(dir_path),key = lambda x:int(x.split('/')[-1].split('.')[0][4:]))
window_list = [60, 4, 8, 12, 16, 20]
columns = ["idx"] + ["window"+str(window_list[i//2])+"_binary" if i % 2 == 0 else "window"+str(window_list[i//2])+"_sum"
                     for i in range(len(window_list) * 2)]

result_df = pd.DataFrame(columns=columns)

for subj in subj_list:
    pred = pd.read_csv(subj)
    row = [os.path.basename(subj)[:-4]]
    for window_size in window_list:
        step = window_size // 2
        decision_count = 0
        binary_True = 0
        sum_True = 0
            
        for trial in range(0,len(pred),30): #trial마다 900개를 30개씩
            temp = pred[trial:trial+30]
            temp_len = len(temp)

            for idx in range(0, temp_len - step + 1):
                attn = temp["True_pred"][idx:idx + step]
                unattn = temp["False_pred"][idx:idx + step]

                result = attn > unattn
                if result.sum() > step // 2:
                    binary_True += 1

                if attn.sum() > unattn.sum():
                    sum_True += 1
                decision_count += 1
        row.extend([binary_True/decision_count, sum_True/decision_count])
    row_df = pd.DataFrame([row], columns=columns)
    result_df = pd.concat([result_df, row_df])
result_df.to_csv(f"./results_G/result_final_{dir_path.split('/')[-2][:-13]}.csv", index=False)
