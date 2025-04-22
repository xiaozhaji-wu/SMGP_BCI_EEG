#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import pickle
from generate_Func import generate_multi_data
from SIM_generate import I, J, L, T, E, Mu_1_true, Mu_0_true, C_t_true, C_e_true, s1, s0, gamma, L_1, L_0, Psi_1, Psi_0, V_Psi_1, V_Psi_0
from model_Func import model, run_mcmc
import matplotlib.pyplot as plt


# define the parser
# seed = int(sys.argv[1])
seed = 0

file_path = f'E:/MSPH/EEG methodology/Advanced EEG Code/Code for GitHub/SMGP_BCI_EEG/SIM_multi/replication_{seed}'
if not os.path.exists(file_path):
    os.makedirs(file_path)

# generate simulation signals
train_name = f"{file_path}/train_data_L_{L}_I_{I}_{seed}.csv"
test_name = f"{file_path}/test_data_L_{L}_I_{I}_{seed}.csv"

if not os.path.exists(train_name):
    train_data = generate_multi_data(Mu_1_true, Mu_0_true, C_t_true, C_e_true, L, I, J, T, E, seed=seed)
    train_data.to_csv(f"{file_path}/train_data_L_{L}_I_{I}_{seed}.csv", index=False)
else:
    train_data = pd.read_csv(train_name)

if not os.path.exists(test_name):
    test_data = generate_multi_data(Mu_1_true, Mu_0_true, C_t_true, C_e_true, L, I + 2, J, T, E, seed=(seed + 100))
    test_data.to_csv(f"{file_path}/test_data_L_{L}_I_{I}_{seed}.csv", index=False)
else:
    test_data = pd.read_csv(test_name)

X1_df = train_data[train_data['Y'] == 1].drop(columns=['Y', 'Letter', 'Sequence', 'Flash'])
X1_samples_per_channel = len(X1_df) // E
X1 = X1_df.iloc[:, :-1].values.reshape(X1_samples_per_channel, E, -1)

X0_df = train_data[train_data['Y'] == 0].drop(columns=['Y', 'Letter', 'Sequence', 'Flash'])
X0_samples_per_channel = len(X0_df) // E
X0 = X0_df.iloc[:, :-1].values.reshape(X0_samples_per_channel, E, -1)

N1 = X1_samples_per_channel
N0 = X0_samples_per_channel

if __name__ == "__main__":
    samples = run_mcmc(model, X1, X0, T, E, N1, N0, s1, s0, gamma, L_1, L_0, Psi_1, Psi_0, V_Psi_1, V_Psi_0, seed)

    sample_dict = {}

    for param_name, param_samples in samples.items():
        # check the shape of the samples
        print(f"Parameter: {param_name}, Before Shape: {param_samples.shape}")

        param_samples_array = param_samples

        chain_1_samples = param_samples_array[::2]
        chain_2_samples = param_samples_array[1::2]

        combined_samples = np.stack([chain_1_samples, chain_2_samples], axis=1)
        sample_dict[param_name] = combined_samples

    # check the shape of the transformed samples dictionary
    for param_name, param_samples in sample_dict.items():
        print(f"Parameter: {param_name}, After Shape: {param_samples.shape}")

    # save the samples
    with open(f'{file_path}/pyro_samples_L_{L}_I_{I}_{seed}.pkl', 'wb') as f:
        pickle.dump(sample_dict, f)
