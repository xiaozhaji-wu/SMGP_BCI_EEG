import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from SIM_generate import (T, L, I, E, Beta_1_1_true, Beta_0_1_true, Beta_1_2_true, Beta_0_2_true,
                          zeta_1_true, zeta_2_true, psi_1_true, psi_2_true, sigma_rho_sq_1_true, sigma_rho_sq_2_true,
                          L_1, L_0, Psi_1, Psi_0, V_Psi_1, V_Psi_0)
from multi_visual_Func import predict
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

matplotlib.use('Agg')

replication = 1

# get the true values
Beta_true = {"Beta_1_1": Beta_1_1_true, "Beta_0_1": Beta_0_1_true,
             "Beta_1_2": Beta_1_2_true, "Beta_0_2": Beta_0_2_true}

zeta_true = {"zeta_1": zeta_1_true, "zeta_2": zeta_2_true}

Psi = {"Psi_1": Psi_1, "Psi_0": Psi_0}

rho_t_list = []
rho_e_list = []

sigma_rho_sq_dict = {}
for e in range(1, E + 1):
    sigma_rho_sq_dict[f'sigma_rho_sq_{e}_list'] = []

theta_dict = {}
alpha_dict = {}
beta_dict = {}
psi_dict = {}
for s in range(2):
    for e in range(1, E + 1):
        theta_dict[f'theta_{s}_{e}_list'] = []
        alpha_dict[f'alpha_{s}_{e}_list'] = []
        beta_dict[f'beta_{s}_{e}_list'] = []
        psi_dict[f'psi_{s}_{e}_list'] = []

zeta_dict = {}
for e in range(1, E + 1):
    zeta_dict[f'zeta_{e}_list'] = []

train_acc_list = []
test_acc_list = []

for rep in tqdm(range(replication)):

    # load the data
    file_path = f'E:/MSPH/EEG methodology/Advanced EEG Code/Code for GitHub/SMGP_BCI_EEG/SIM_multi/replication_{rep}'

    # plot_path = f'{file_path}/hard_plots/L_{L}_I_{I}'
    R_path = f'{file_path}/R_plots/L_{L}_I_{I}'
    # if not os.path.exists(plot_path):
    #     os.makedirs(plot_path)
    if not os.path.exists(R_path):
        os.makedirs(R_path)

    train_data = pd.read_csv(f'{file_path}/train_data_L_{L}_I_{I}_{rep}.csv')
    training = train_data.copy()
    test_data = pd.read_csv(f'{file_path}/test_data_L_{L}_I_{I}_{rep}.csv')
    testing = test_data.copy()

    with open(f'{file_path}/pyro_samples_L_{L}_I_{I}_{rep}.pkl', 'rb') as f:
        pyro_samples = pickle.load(f)

    # get the samples
    rho_t = pyro_samples['rho_t']
    rho_t_list.append(rho_t)

    rho_e = pyro_samples['rho_e']
    rho_e_list.append(rho_e)

    sigma_rho = {}
    sigma_rho_sq = {}
    for e in range(1, E + 1):
        sigma_rho[f'sigma_rho_{e}'] = pyro_samples[f'sigma_rho_{e}']

        sigma_rho_sq[f'sigma_rho_sq_{e}'] = sigma_rho[f'sigma_rho_{e}'] ** 2
        sigma_rho_sq_dict[f'sigma_rho_sq_{e}_list'].append(sigma_rho_sq[f'sigma_rho_sq_{e}'])

    omega = {}
    for e in range(1, E + 1):
        omega[f'omega_{e}'] = pyro_samples[f'omega_{e}']

    zeta = {}
    for e in range(1, E + 1):
        zeta[f'zeta_{e}'] = norm.cdf(omega[f'omega_{e}'])
        zeta_dict[f'zeta_{e}_list'].append(zeta[f'zeta_{e}'])

    theta = {}
    alpha = {}
    psi = {}
    for s in range(2):
        for e in range(1, E + 1):
            psi[f'psi_{s}_{e}'] = pyro_samples[f'psi_{s}_{e}']
            psi_dict[f'psi_{s}_{e}_list'].append(psi[f'psi_{s}_{e}'])

            theta[f'theta_{s}_{e}'] = pyro_samples[f'theta_{s}_{e}']
            theta_dict[f'theta_{s}_{e}_list'].append(theta[f'theta_{s}_{e}'])

            alpha[f'alpha_{s}_{e}'] = psi[f'psi_{s}_{e}'][:, :, np.newaxis] * np.einsum('ijk,kl->ijl',
                                                                                        theta[f'theta_{s}_{e}'],
                                                                                        np.transpose(Psi[f'Psi_{s}']))
            alpha_dict[f'alpha_{s}_{e}_list'].append(alpha[f'alpha_{s}_{e}'])

    beta = {}
    for s in range(2):
        for e in range(1, E + 1):
            if s == 0:
                beta[f'beta_{s}_{e}'] = alpha[f'alpha_{s}_{e}']
            elif s == 1:
                beta[f'beta_{s}_{e}'] = zeta[f'zeta_{e}'] * alpha[f'alpha_{s}_{e}'] + (1 - zeta[f'zeta_{e}']) * \
                                        alpha[f'alpha_{0}_{e}']

    C_t = np.zeros((1000, 2, T, T))
    C_e = np.zeros((1000, 2, E, E))
    for i in range(1000):
        for j in range(2):
            for k in range(T):
                for l in range(T):
                    if k == l:
                        C_t[i, j, k, l] = 1
                    else:
                        C_t[i, j, k, l] = (rho_t[i, j] ** abs(k - l))

            sigma_rho_diag = np.diag([sigma_rho[f'sigma_rho_{e}'][i, j] for e in range(1, E + 1)])
            for m in range(E):
                for n in range(E):
                    if m == n:
                        C_e[i, j, m, n] = 1
                    else:
                        C_e[i, j, m, n] = rho_e[i, j]
            C_e[i, j] = np.dot(np.dot(sigma_rho_diag, C_e[i, j]), sigma_rho_diag)

    # calculate the mean of each parameter
    rho_t_mean = np.mean(rho_t, axis=1)

    rho_e_mean = np.mean(rho_e, axis=1)

    sigma_rho_sq_mean = {}

    for e in range(1, E + 1):
        sigma_rho_sq_mean[f'sigma_rho_sq_{e}'] = np.mean(sigma_rho_sq[f'sigma_rho_sq_{e}'], axis=1)

    alpha_mean = {}
    alpha_sd = {}
    alpha_range = {}
    psi_mean = {}
    for s in range(2):
        for e in range(1, E + 1):
            alpha_mean[f'alpha_{s}_{e}'] = np.mean(alpha[f'alpha_{s}_{e}'], axis=(0, 1))
            alpha_sd[f'alpha_{s}_{e}'] = np.std(alpha[f'alpha_{s}_{e}'], axis=(0, 1))
            alpha_range[f'alpha_{s}_{e}'] = np.vstack((alpha_mean[f'alpha_{s}_{e}'] - 1.96 * alpha_sd[f'alpha_{s}_{e}'],
                                                       alpha_mean[f'alpha_{s}_{e}'] + 1.96 * alpha_sd[
                                                           f'alpha_{s}_{e}']))

            psi_mean[f'psi_{s}_{e}'] = np.mean(psi[f'psi_{s}_{e}'], axis=1)

    zeta_mean = {}
    zeta_sd = {}
    zeta_range = {}
    beta_mean = {}
    beta_sd = {}
    beta_range = {}
    for e in range(1, E + 1):
        if e == 1:
            threshold = 0.8
        elif e == 2:
            threshold = 0.8
        zeta_mean[f'zeta_{e}'] = np.mean(zeta[f'zeta_{e}'], axis=(0, 1))
        zeta_sd[f'zeta_{e}'] = np.std(zeta[f'zeta_{e}'], axis=(0, 1))
        zeta_range[f'zeta_{e}'] = np.vstack((zeta_mean[f'zeta_{e}'] - 1.96 * zeta_sd[f'zeta_{e}'],
                                             zeta_mean[f'zeta_{e}'] + 1.96 * zeta_sd[f'zeta_{e}']))

        time_points = zeta_mean[f'zeta_{e}'] < threshold
        time_points = time_points[np.newaxis, np.newaxis, :]
        beta[f"beta_1_{e}"] = np.where(time_points, beta[f"beta_0_{e}"], alpha[f"alpha_1_{e}"])

        for s in range(2):
            # if s == 0:
            #     beta_mean[f'beta_{s}_{e}'] = alpha_mean[f'alpha_{s}_{e}']
            # elif s == 1:
            #     beta_mean[f'beta_{s}_{e}'] = np.where(time_points, alpha_mean[f'alpha_{0}_{e}'],
            #                                           zeta_mean[f'zeta_{e}'] * alpha_mean[f'alpha_{s}_{e}'] + \
            #                                           (1 - zeta_mean[f'zeta_{e}']) * alpha_mean[f'alpha_{0}_{e}'])
            #     beta_mean[f'beta_{s}_{e}'] = beta_mean[f'beta_{s}_{e}'].flatten()
            beta_mean[f'beta_{s}_{e}'] = np.mean(beta[f'beta_{s}_{e}'], axis=(0, 1))
            beta_sd[f'beta_{s}_{e}'] = np.std(beta[f'beta_{s}_{e}'], axis=(0, 1))
            beta_range[f'beta_{s}_{e}'] = np.vstack((beta_mean[f'beta_{s}_{e}'] - 1.96 * beta_sd[f'beta_{s}_{e}'],
                                                         beta_mean[f'beta_{s}_{e}'] + 1.96 * beta_sd[f'beta_{s}_{e}']))
            beta_dict[f'beta_{s}_{e}_list'].append(beta[f'beta_{s}_{e}'])

    Mu_mean = {}
    for s in range(2):
        Mu_mean[f'Mu_{s}'] = np.vstack([beta_mean[f'beta_{s}_{e}'] for e in range(1, E + 1)])

    C_t_mean = np.mean(C_t, axis=(0, 1))
    C_e_mean = np.mean(C_e, axis=(0, 1))

    train_acc, train_prob = predict(training, Mu_mean["Mu_1"], Mu_mean["Mu_0"], C_e_mean, C_t_mean, T, E)
    test_acc, test_prob = predict(testing, Mu_mean["Mu_1"], Mu_mean["Mu_0"], C_e_mean, C_t_mean, T, E)
    train_array = np.array(train_acc)
    test_array = np.array(test_acc)
    np.savetxt(f'{R_path}/train_acc.csv', train_array, delimiter=',')
    np.savetxt(f'{R_path}/test_acc.csv', test_array, delimiter=',')
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    zeta_dfs = []
    for e in range(1, E + 1):
        zeta_df = pd.DataFrame({
            'Time': np.linspace(0, 1000, T),
            'Zeta_mean': zeta_mean[f'zeta_{e}'],
            'Zeta_min': zeta_range[f'zeta_{e}'][0],
            'Zeta_max': zeta_range[f'zeta_{e}'][1],
            'Zeta_true': zeta_true[f'zeta_{e}'].flatten(),
            'Channel': f'Channel {e}'
        })
        zeta_dfs.append(zeta_df)
        zeta_df.to_csv(f'{R_path}/zeta_{e}.csv', index=False)

    beta_dfs = []
    for e in range(1, E + 1):
        beta_df = pd.DataFrame({
            'Time': np.linspace(0, 1000, T),
            'Beta_0_mean': beta_mean[f'beta_0_{e}'],
            'Beta_0_min': beta_range[f'beta_0_{e}'][0],
            'Beta_0_max': beta_range[f'beta_0_{e}'][1],
            'Beta_1_mean': beta_mean[f'beta_1_{e}'],
            'Beta_1_min': beta_range[f'beta_1_{e}'][0],
            'Beta_1_max': beta_range[f'beta_1_{e}'][1],
            'Beta_0_true': Beta_true[f'Beta_0_{e}'].flatten(),
            'Beta_1_true': Beta_true[f'Beta_1_{e}'].flatten(),
            'Channel': f'Channel {e}'
        })
        beta_dfs.append(beta_df)
        beta_df.to_csv(f'{R_path}/beta_{e}.csv', index=False)

#     # visualize alpha and beta
#     for e in range(1, E + 1):
#         fig, ax = plt.subplots(1, 2, figsize=(24, 12))
#         ax[0].plot(np.arange(T), alpha_mean[f'alpha_0_{e}'], label=f'Alpha_0', color='blue')
#         ax[0].fill_between(np.arange(T), alpha_range[f'alpha_0_{e}'][0], alpha_range[f'alpha_0_{e}'][1], alpha=0.2,
#                            color='blue')
#         ax[0].plot(np.arange(T), alpha_mean[f'alpha_1_{e}'], label=f'Alpha_1', color='red')
#         ax[0].fill_between(np.arange(T), alpha_range[f'alpha_1_{e}'][0], alpha_range[f'alpha_1_{e}'][1], alpha=0.2,
#                            color='red')
#         ax[0].plot(np.arange(T), Beta_true[f'Beta_0_{e}'], label=f'Beta_0_true', color='blue', linestyle='--')
#         ax[0].plot(np.arange(T), Beta_true[f'Beta_1_{e}'], label=f'Beta_1_true', color='red', linestyle='--')
#         ax[0].legend()
#
#         ax[1].plot(np.arange(T), beta_mean[f'beta_0_{e}'], label=f'Beta_0', color='blue')
#         ax[1].fill_between(np.arange(T), beta_range[f'beta_0_{e}'][0], beta_range[f'beta_0_{e}'][1], alpha=0.2,
#                            color='blue')
#         ax[1].plot(np.arange(T), beta_mean[f'beta_1_{e}'], label=f'Beta_1', color='red')
#         ax[1].fill_between(np.arange(T), beta_range[f'beta_1_{e}'][0], beta_range[f'beta_1_{e}'][1], alpha=0.2,
#                            color='red')
#         ax[1].plot(np.arange(T), Beta_true[f'Beta_0_{e}'], label=f'Beta_0_true', color='blue', linestyle='--')
#         ax[1].plot(np.arange(T), Beta_true[f'Beta_1_{e}'], label=f'Beta_1_true', color='red', linestyle='--')
#         ax[1].legend()
#         ax[0].set_title(f'Alpha for Channel {e}')
#         ax[1].set_title(f'Beta for Channel {e}')
#         plt.savefig(f'{plot_path}/Alpha and Beta for Channel {e}.png')
#         plt.close()
#
#     for e in range(1, E + 1):
#         plt.figure(figsize=(12, 6))
#         plt.plot(np.arange(T), zeta_mean[f'zeta_{e}'])
#         plt.fill_between(np.arange(T), zeta_range[f'zeta_{e}'][0], zeta_range[f'zeta_{e}'][1], alpha=0.2)
#         plt.plot(np.arange(T), zeta_true[f'zeta_{e}'], linestyle='--')
#         plt.ylim(0, 1)
#         plt.title(f'Zeta for Channel {e}')
#         plt.savefig(f'{plot_path}/Zeta for Channel {e}.png')
#         plt.close()
#
#     # visualize train and test accuracy
#     fig, ax = plt.subplots(1, 2, figsize=(12, 6))
#     ax[0].plot(np.arange(1, I + 1), train_acc)
#     ax[0].set_title('Train Accuracy')
#     ax[0].set_xlabel('Number of Sequence')
#     ax[0].set_ylabel('Accuracy')
#     ax[0].set_ylim(0, 1)
#
#     ax[1].plot(np.arange(1, I + 11), test_acc)
#     ax[1].set_title('Test Accuracy')
#     ax[1].set_xlabel('Number of Sequence')
#     ax[1].set_ylabel('Accuracy')
#     ax[1].set_ylim(0, 1)
#     plt.savefig(f'{plot_path}/Train and Test Accuracy.png')
#     plt.close()
#
# alpha_array = {}
# beta_array = {}
# for s in range(2):
#     for e in range(1, E + 1):
#         alpha_array[f'alpha_{s}_{e}'] = np.array(alpha_dict[f'alpha_{s}_{e}_list'])
#         beta_array[f'beta_{s}_{e}'] = np.array(beta_dict[f'beta_{s}_{e}_list'])
#
# zeta_array = {}
# for e in range(1, E + 1):
#     zeta_array[f'zeta_{e}'] = np.array(zeta_dict[f'zeta_{e}_list'])
#
# train_acc_array = np.array(train_acc_list)
# test_acc_array = np.array(test_acc_list)
#
# alpha_mean = {}
# alpha_sd = {}
# alpha_range = {}
# beta_mean = {}
# beta_sd = {}
# beta_range = {}
#
# for s in range(2):
#     for e in range(1, E + 1):
#         alpha_mean[f'alpha_{s}_{e}'] = np.mean(alpha_array[f'alpha_{s}_{e}'], axis=(0, 1, 2))
#         alpha_sd[f'alpha_{s}_{e}'] = np.std(alpha_array[f'alpha_{s}_{e}'], axis=(0, 1, 2))
#         alpha_range[f'alpha_{s}_{e}'] = np.vstack((alpha_mean[f'alpha_{s}_{e}'] - 1.96 * alpha_sd[f'alpha_{s}_{e}'],
#                                                    alpha_mean[f'alpha_{s}_{e}'] + 1.96 * alpha_sd[f'alpha_{s}_{e}']))
#         beta_mean[f'beta_{s}_{e}'] = np.mean(beta_array[f'beta_{s}_{e}'], axis=(0, 1, 2))
#         beta_sd[f'beta_{s}_{e}'] = np.std(beta_array[f'beta_{s}_{e}'], axis=(0, 1, 2))
#         beta_range[f'beta_{s}_{e}'] = np.vstack((beta_mean[f'beta_{s}_{e}'] - 1.96 * beta_sd[f'beta_{s}_{e}'],
#                                                  beta_mean[f'beta_{s}_{e}'] + 1.96 * beta_sd[f'beta_{s}_{e}']))
#
# zeta_mean = {}
# zeta_sd = {}
# zeta_range = {}
# for e in range(1, E + 1):
#     zeta_mean[f'zeta_{e}'] = np.mean(zeta_array[f'zeta_{e}'], axis=(0, 1, 2))
#     zeta_sd[f'zeta_{e}'] = np.std(zeta_array[f'zeta_{e}'], axis=(0, 1, 2))
#     zeta_range[f'zeta_{e}'] = np.vstack((zeta_mean[f'zeta_{e}'] - 1.96 * zeta_sd[f'zeta_{e}'],
#                                          zeta_mean[f'zeta_{e}'] + 1.96 * zeta_sd[f'zeta_{e}']))
#
# train_acc_mean = np.mean(train_acc_array, axis=0)
# train_acc_sd = np.std(train_acc_array, axis=0)
# train_acc_range = np.vstack((train_acc_mean - 1.96 * train_acc_sd, train_acc_mean + 1.96 * train_acc_sd))
#
# test_acc_mean = np.mean(test_acc_array, axis=0)
# test_acc_sd = np.std(test_acc_array, axis=0)
# test_acc_range = np.vstack((test_acc_mean - 1.96 * test_acc_sd, test_acc_mean + 1.96 * test_acc_sd))
#
# # visualize the overall mean of each parameter
#
# overall_plot_path = f'SIM_multi/hard_plots/L_{L}_I_{I}'
# overall_R_path = f'SIM_multi/R_plots/L_{L}_I_{I}'
# if not os.path.exists(overall_plot_path):
#     os.makedirs(overall_plot_path)
# if not os.path.exists(overall_R_path):
#     os.makedirs(overall_R_path)
#
# np.savetxt(f'{overall_R_path}/train_acc.csv', train_acc_mean, delimiter=',')
# np.savetxt(f'{overall_R_path}/test_acc.csv', test_acc_mean, delimiter=',')
#
# zeta_dfs = []
# for e in range(1, E + 1):
#     zeta_df = pd.DataFrame({
#         'Time': np.linspace(0, 1000, T),
#         'Zeta_mean': zeta_mean[f'zeta_{e}'],
#         'Zeta_min': zeta_range[f'zeta_{e}'][0],
#         'Zeta_max': zeta_range[f'zeta_{e}'][1],
#         'Zeta_true': zeta_true[f'zeta_{e}'].flatten(),
#         'Channel': f'Channel {e}'
#     })
#     zeta_dfs.append(zeta_df)
#     zeta_df.to_csv(f'{overall_R_path}/zeta_{e}.csv', index=False)
#
# beta_dfs = []
# for e in range(1, E + 1):
#     beta_df = pd.DataFrame({
#         'Time': np.linspace(0, 1000, T),
#         'Beta_0_mean': beta_mean[f'beta_0_{e}'],
#         'Beta_0_min': beta_range[f'beta_0_{e}'][0],
#         'Beta_0_max': beta_range[f'beta_0_{e}'][1],
#         'Beta_1_mean': beta_mean[f'beta_1_{e}'],
#         'Beta_1_min': beta_range[f'beta_1_{e}'][0],
#         'Beta_1_max': beta_range[f'beta_1_{e}'][1],
#         'Beta_0_true': Beta_true[f'Beta_0_{e}'].flatten(),
#         'Beta_1_true': Beta_true[f'Beta_1_{e}'].flatten(),
#         'Channel': f'Channel {e}'
#     })
#     beta_dfs.append(beta_df)
#     beta_df.to_csv(f'{overall_R_path}/beta_{e}.csv', index=False)
#
#
# # visualize alpha and beta
# for e in range(1, E + 1):
#     fig, ax = plt.subplots(1, 2, figsize=(24, 12))
#     ax[0].plot(np.arange(T), alpha_mean[f'alpha_0_{e}'], label=f'Alpha_0', color='blue')
#     ax[0].fill_between(np.arange(T), alpha_range[f'alpha_0_{e}'][0], alpha_range[f'alpha_0_{e}'][1], alpha=0.2,
#                        color='blue')
#     ax[0].plot(np.arange(T), alpha_mean[f'alpha_1_{e}'], label=f'Alpha_1', color='red')
#     ax[0].fill_between(np.arange(T), alpha_range[f'alpha_1_{e}'][0], alpha_range[f'alpha_1_{e}'][1], alpha=0.2,
#                        color='red')
#     ax[0].plot(np.arange(T), Beta_true[f'Beta_0_{e}'], label=f'Beta_0_true', color='blue', linestyle='--')
#     ax[0].plot(np.arange(T), Beta_true[f'Beta_1_{e}'], label=f'Beta_1_true', color='red', linestyle='--')
#     ax[0].legend()
#
#     ax[1].plot(np.arange(T), beta_mean[f'beta_0_{e}'], label=f'Beta_0', color='blue')
#     ax[1].fill_between(np.arange(T), beta_range[f'beta_0_{e}'][0], beta_range[f'beta_0_{e}'][1], alpha=0.2,
#                        color='blue')
#     ax[1].plot(np.arange(T), beta_mean[f'beta_1_{e}'], label=f'Beta_1', color='red')
#     ax[1].fill_between(np.arange(T), beta_range[f'beta_1_{e}'][0], beta_range[f'beta_1_{e}'][1], alpha=0.2, color='red')
#     ax[1].plot(np.arange(T), Beta_true[f'Beta_0_{e}'], label=f'Beta_0_true', color='blue', linestyle='--')
#     ax[1].plot(np.arange(T), Beta_true[f'Beta_1_{e}'], label=f'Beta_1_true', color='red', linestyle='--')
#     ax[1].legend()
#     ax[0].set_title(f'Alpha for Channel {e}')
#     ax[1].set_title(f'Beta for Channel {e}')
#     plt.savefig(f'{overall_plot_path}/Alpha and Beta for Channel {e}.png')
#     plt.close()
#
# # visualize zeta
# for e in range(1, E + 1):
#     plt.figure(figsize=(12, 6))
#     plt.plot(np.arange(T), zeta_mean[f'zeta_{e}'])
#     plt.fill_between(np.arange(T), zeta_range[f'zeta_{e}'][0], zeta_range[f'zeta_{e}'][1], alpha=0.2)
#     plt.plot(np.arange(T), zeta_true[f'zeta_{e}'], linestyle='--')
#     plt.ylim(0, 1)
#     plt.title(f'Zeta for Channel {e}')
#     plt.savefig(f'{overall_plot_path}/Zeta for Channel {e}.png')
#     plt.close()
#
# # visualize train and test accuracy
# fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# ax[0].plot(np.arange(1, I + 1), train_acc_mean)
# ax[0].fill_between(np.arange(1, I + 1), train_acc_range[0], train_acc_range[1], alpha=0.2)
# ax[0].set_title('Train Accuracy')
# ax[0].set_xlabel('Number of Sequence')
# ax[0].set_ylabel('Accuracy')
# ax[0].set_ylim(0, 1)
#
# ax[1].plot(np.arange(1, I + 11), test_acc_mean)
# ax[1].fill_between(np.arange(1, I + 11), test_acc_range[0], test_acc_range[1], alpha=0.2)
# ax[1].set_title('Test Accuracy')
# ax[1].set_xlabel('Number of Sequence')
# ax[1].set_ylabel('Accuracy')
# ax[1].set_ylim(0, 1)
# plt.savefig(f'{overall_plot_path}/Train and Test Accuracy.png')
# plt.close()
