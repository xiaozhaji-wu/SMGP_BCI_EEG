import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from SIM_generate import (T, L, I, E, Beta_1_1_true, Beta_0_1_true, Beta_1_2_true, Beta_0_2_true,
                          zeta_1_true, zeta_2_true, psi_1_true, psi_2_true, sigma_rho_sq_1_true, sigma_rho_sq_2_true,
                          L_1, L_0, Psi_1, Psi_0, V_Psi_1, V_Psi_0)
from single_visual_Func import predict
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

matplotlib.use('Agg')

replication = 100

channel = 1

# get the true values
Beta_true = {"Beta_1_1": Beta_1_1_true, "Beta_0_1": Beta_0_1_true,
             "Beta_1_2": Beta_1_2_true, "Beta_0_2": Beta_0_2_true}

zeta_true = {"zeta_1": zeta_1_true, "zeta_2": zeta_2_true}

Psi = {"Psi_1": Psi_1, "Psi_0": Psi_0}

psi_true = {"psi_1": psi_1_true, "psi_2": psi_2_true}

sigma_rho_sq_true = {"sigma_rho_sq_1": sigma_rho_sq_1_true, "sigma_rho_sq_2": sigma_rho_sq_2_true}

rho_t_list = []

sigma_rho_sq_list = []

theta_dict = {}
alpha_dict = {}
beta_dict = {}
psi_dict = {}
for s in range(2):
    theta_dict[f'theta_{s}_list'] = []
    alpha_dict[f'alpha_{s}_list'] = []
    beta_dict[f'beta_{s}_list'] = []
    psi_dict[f'psi_{s}_list'] = []

rho_omega_list = []
psi_omega_list = []
omega_list = []
zeta_list = []

train_acc_list = []
test_acc_list = []

for rep in tqdm(range(replication)):

    # load the data
    file_path = f'SIM_single/Channel_{channel}/replication_{rep}'

    plot_path = f'{file_path}/hard_plots/L_{L}_I_{I}'
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    train_data = pd.read_csv(f'{file_path}/train_data_L_{L}_I_{I}_{rep}.csv')
    training = train_data[train_data['Channel'] == channel].drop(columns=['Channel'])
    test_data = pd.read_csv(f'{file_path}/test_data_L_{L}_I_{I}_{rep}.csv')
    testing = test_data[test_data['Channel'] == channel].drop(columns=['Channel'])

    with open(f'{file_path}/pyro_samples_L_{L}_I_{I}_{rep}.pkl', 'rb') as f:
        pyro_samples = pickle.load(f)

    # get the samples
    rho_t = pyro_samples['rho_t']
    rho_t_list.append(rho_t)

    sigma_rho = pyro_samples[f'sigma_rho']
    sigma_rho_sq = sigma_rho ** 2
    sigma_rho_sq_list.append(sigma_rho_sq)

    theta = {}
    alpha = {}
    psi = {}
    for s in range(2):
        psi[f'psi_{s}'] = pyro_samples[f'psi_{s}']
        psi_dict[f'psi_{s}_list'].append(psi[f'psi_{s}'])

        theta[f'theta_{s}'] = pyro_samples[f'theta_{s}']
        theta_dict[f'theta_{s}_list'].append(theta[f'theta_{s}'])

        alpha[f'alpha_{s}'] = psi[f'psi_{s}'][:, :, np.newaxis] * np.einsum('ijk,kl->ijl', theta[f'theta_{s}'],
                                                                            np.transpose(Psi[f'Psi_{s}']))
        alpha_dict[f'alpha_{s}_list'].append(alpha[f'alpha_{s}'])

    omega = pyro_samples[f'omega']
    omega_list.append(omega)

    zeta = norm.cdf(omega)
    zeta_list.append(zeta)

    beta = {}
    for s in range(2):
        if s == 0:
            beta[f'beta_{s}'] = alpha[f'alpha_{s}']
        elif s == 1:
            beta[f'beta_{s}'] = zeta * alpha[f'alpha_{s}'] + (1 - zeta) * alpha[f'alpha_{0}']

    C_t = np.zeros((1000, 2, T, T))
    for i in range(1000):
        for j in range(2):
            for k in range(T):
                for l in range(T):
                    if k == l:
                        C_t[i, j, k, l] = 1
                    else:
                        C_t[i, j, k, l] = (rho_t[i, j] ** abs(k - l))

    # calculate the mean of each parameter
    rho_t_mean = np.mean(rho_t, axis=1)

    sigma_rho_sq_mean = np.mean(sigma_rho_sq, axis=1)

    alpha_mean = {}
    alpha_sd = {}
    alpha_range = {}
    psi_mean = {}
    for s in range(2):
        alpha_mean[f'alpha_{s}'] = np.mean(alpha[f'alpha_{s}'], axis=(0, 1))
        alpha_sd[f'alpha_{s}'] = np.std(alpha[f'alpha_{s}'], axis=(0, 1))
        alpha_range[f'alpha_{s}'] = np.vstack((alpha_mean[f'alpha_{s}'] - 1.96 * alpha_sd[f'alpha_{s}'],
                                               alpha_mean[f'alpha_{s}'] + 1.96 * alpha_sd[f'alpha_{s}']))
        psi_mean[f'psi_{s}'] = np.mean(psi[f'psi_{s}'], axis=(0, 1))

    zeta_mean = {}
    zeta_sd = {}
    zeta_range = {}
    beta_mean = {}
    beta_sd = {}
    beta_range = {}
    threshold = 0.8
    zeta_mean = np.mean(zeta, axis=(0, 1))
    zeta_sd = np.std(zeta, axis=(0, 1))
    zeta_range = np.vstack((zeta_mean - 1.96 * zeta_sd, zeta_mean + 1.96 * zeta_sd))

    time_points = zeta_mean < threshold
    time_points = time_points[np.newaxis, np.newaxis, :]
    beta[f"beta_1"] = np.where(time_points, beta[f"beta_0"], alpha[f"alpha_1"])

    for s in range(2):
        beta_mean[f'beta_{s}'] = np.mean(beta[f'beta_{s}'], axis=(0, 1))
        beta_sd[f'beta_{s}'] = np.std(beta[f'beta_{s}'], axis=(0, 1))
        beta_range[f'beta_{s}'] = np.vstack((beta_mean[f'beta_{s}'] - 1.96 * beta_sd[f'beta_{s}'],
                                             beta_mean[f'beta_{s}'] + 1.96 * beta_sd[f'beta_{s}']))
        beta_dict[f'beta_{s}_list'].append(beta[f'beta_{s}'])

    C_t_mean = np.mean(C_t, axis=(0, 1))

    train_acc, train_prob = predict(training, beta_mean["beta_1"], beta_mean["beta_0"], C_t_mean, T)
    test_acc, test_prob = predict(testing, beta_mean["beta_1"], beta_mean["beta_0"], C_t_mean, T)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    # visualize alpha and beta
    fig, ax = plt.subplots(1, 2, figsize=(24, 12))
    ax[0].plot(np.arange(T), alpha_mean[f'alpha_0'], label=f'Alpha_0', color='blue')
    ax[0].fill_between(np.arange(T), alpha_range[f'alpha_0'][0], alpha_range[f'alpha_0'][1], alpha=0.2,
                       color='blue')
    ax[0].plot(np.arange(T), alpha_mean[f'alpha_1'], label=f'Alpha_1', color='red')
    ax[0].fill_between(np.arange(T), alpha_range[f'alpha_1'][0], alpha_range[f'alpha_1'][1], alpha=0.2,
                       color='red')
    ax[0].plot(np.arange(T), Beta_true[f'Beta_0_{channel}'], label=f'Beta_0_true', color='blue', linestyle='--')
    ax[0].plot(np.arange(T), Beta_true[f'Beta_1_{channel}'], label=f'Beta_1_true', color='red', linestyle='--')
    ax[0].legend()

    ax[1].plot(np.arange(T), beta_mean[f'beta_0'], label=f'Beta_0', color='blue')
    ax[1].fill_between(np.arange(T), beta_range[f'beta_0'][0], beta_range[f'beta_0'][1], alpha=0.2,
                       color='blue')
    ax[1].plot(np.arange(T), beta_mean[f'beta_1'], label=f'Beta_1', color='red')
    ax[1].fill_between(np.arange(T), beta_range[f'beta_1'][0], beta_range[f'beta_1'][1], alpha=0.2,
                       color='red')
    ax[1].plot(np.arange(T), Beta_true[f'Beta_0_{channel}'], label=f'Beta_0_true', color='blue', linestyle='--')
    ax[1].plot(np.arange(T), Beta_true[f'Beta_1_{channel}'], label=f'Beta_1_true', color='red', linestyle='--')
    ax[1].legend()
    ax[0].set_title(f'Alpha for Channel {channel}')
    ax[1].set_title(f'Beta for Channel {channel}')
    plt.savefig(f'{plot_path}/Alpha and Beta for Channel {channel}.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(T), zeta_mean)
    plt.fill_between(np.arange(T), zeta_range[0], zeta_range[1], alpha=0.2)
    plt.plot(np.arange(T), zeta_true[f'zeta_{channel}'], linestyle='--')
    plt.ylim(0, 1)
    plt.title(f'Zeta for Channel {channel}')
    plt.savefig(f'{plot_path}/Zeta for Channel {channel}.png')
    plt.close()

    # visualize train and test accuracy
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(np.arange(1, I + 1), train_acc)
    ax[0].set_title('Train Accuracy')
    ax[0].set_xlabel('Number of Sequence')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim(0, 1)

    ax[1].plot(np.arange(1, I + 1), test_acc)
    ax[1].set_title('Test Accuracy')
    ax[1].set_xlabel('Number of Sequence')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_ylim(0, 1)
    plt.savefig(f'{plot_path}/Train and Test Accuracy.png')
    plt.close()
