#!/usr/bin/env python3

import os
import numpy as np
import scipy.io as sio
import pickle
from EEG_generate import T, E, s1, s0, gamma, L_1, L_0, Psi_1, Psi_0, V_Psi_1, V_Psi_0
from model_Func import model_ref, run_mcmc

# define the parser
# seed = int(sys.argv[1])
seed = 18

if seed == 0:
    person = "K151"
elif seed == 1:
    person = "K114"
elif seed == 2:
    person = "K106"
elif seed == 3:
    person = "K111"
elif seed == 4:
    person = "K112"
elif seed == 5:
    person = "K160"
elif seed == 6:
    person = "K172"
elif seed == 7:
    person = "K183"
elif seed == 8:
    person = "K145"
elif seed == 9:
    person = "K113"
elif seed == 10:
    person = "K121"
elif seed == 11:
    person = "K118"
elif seed == 12:
    person = "K107"
elif seed == 13:
    person = "K190"
elif seed == 14:
    person = "K191"
elif seed == 15:
    person = "K159"
elif seed == 16:
    person = "K185"
elif seed == 17:
    person = "K184"
elif seed == 18:
    person = "K178"
elif seed == 19:
    person = "K154"
elif seed == 20:
    person = "K166"


file_path = f'EEG_multi_ref/{person}'
if not os.path.exists(file_path):
    os.makedirs(file_path)

data = sio.loadmat(f"EEG_multi_ref/{person}/{person}_TRN_xDAWN.mat")
X = data['X']
Y = data['Y']
X1 = X[Y.flatten() == 1]
X0 = X[Y.flatten() == 0]
X1 = X1.reshape(X1.shape[0], E, T)
X0 = X0.reshape(X0.shape[0], E, T)
N1 = X1.shape[0]
N0 = X0.shape[0]

if __name__ == "__main__":
    samples = run_mcmc(model_ref, X1, X0, T, E, N1, N0, s1, s0, gamma, L_1, L_0, Psi_1, Psi_0, V_Psi_1, V_Psi_0, seed)

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
    with open(f'EEG_multi_ref/{person}/pyro_samples_{person}.pkl', 'wb') as f:
        pickle.dump(sample_dict, f)
