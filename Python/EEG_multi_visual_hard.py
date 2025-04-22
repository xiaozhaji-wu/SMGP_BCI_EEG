import numpy as np
import pandas as pd
import scipy.io as sio
import os
import pickle
from EEG_generate import T, E, Psi_1, Psi_0
from EEG_visual_Func import predict
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm

matplotlib.use('Agg')

method = 2
threshold_set = 0.5

Psi = {"Psi_1": Psi_1, "Psi_0": Psi_0}

for seed in range(0, 21):

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

    if method == 1:
        file_path = f'EEG_multi/{person}'
        plot_path = f'EEG_multi/{person}/hard_plots'
        R_path = f'EEG_multi/{person}/R_plots'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        if not os.path.exists(R_path):
            os.makedirs(R_path)
    elif method == 2:
        file_path = f'EEG_multi_ref/{person}'
        plot_path = f'EEG_multi_ref/{person}/plots'
        R_path = f'EEG_multi_ref/{person}/R_plots'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        if not os.path.exists(R_path):
            os.makedirs(R_path)

    train_data = sio.loadmat(f"{file_path}/{person}_TRN_xDAWN.mat")
    training = train_data.copy()
    test_data = sio.loadmat(f"{file_path}/{person}_FRT_xDAWN.mat")
    testing = test_data.copy()

    with open(f"{file_path}/pyro_samples_{person}.pkl", 'rb') as f:
        pyro_samples = pickle.load(f)

    # get the samples
    rho_t = pyro_samples['rho_t']

    rho_e = pyro_samples['rho_e']

    sigma_rho = {}
    sigma_rho_sq = {}
    for e in range(1, E + 1):
        sigma_rho[f'sigma_rho_{e}'] = pyro_samples[f'sigma_rho_{e}']
        sigma_rho_sq[f'sigma_rho_sq_{e}'] = sigma_rho[f'sigma_rho_{e}'] ** 2

    theta = {}
    alpha = {}
    psi = {}
    for s in range(2):
        for e in range(1, E + 1):
            psi[f'psi_{s}_{e}'] = pyro_samples[f'psi_{s}_{e}']

            theta[f'theta_{s}_{e}'] = pyro_samples[f'theta_{s}_{e}']

            alpha[f'alpha_{s}_{e}'] = psi[f'psi_{s}_{e}'][:, :, np.newaxis] * np.einsum('ijk,kl->ijl',
                                                                                        theta[f'theta_{s}_{e}'],
                                                                                        np.transpose(Psi[f'Psi_{s}']))

    if method == 1:
        omega = {}
        for e in range(1, E + 1):
            omega[f'omega_{e}'] = pyro_samples[f'omega_{e}']

        zeta = {}
        for e in range(1, E + 1):
            zeta[f'zeta_{e}'] = norm.cdf(omega[f'omega_{e}'])

        beta = {}
        for s in range(2):
            for e in range(1, E + 1):
                if s == 0:
                    beta[f'beta_{s}_{e}'] = alpha[f'alpha_{s}_{e}']
                elif s == 1:
                    beta[f'beta_{s}_{e}'] = zeta[f'zeta_{e}'] * alpha[f'alpha_{s}_{e}'] + (1 - zeta[f'zeta_{e}']) * \
                                            alpha[f'alpha_{0}_{e}']

    elif method == 2:
        beta = {}
        for s in range(2):
            for e in range(1, E + 1):
                beta[f'beta_{s}_{e}'] = alpha[f'alpha_{s}_{e}']

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

    if method == 1:
        zeta_mean = {}
        zeta_sd = {}
        zeta_range = {}
        beta_mean = {}
        beta_sd = {}
        beta_range = {}
        for e in range(1, E + 1):
            if e == 1:
                threshold = threshold_set
            elif e == 2:
                threshold = threshold_set
            zeta_mean[f'zeta_{e}'] = np.mean(zeta[f'zeta_{e}'], axis=(0, 1))
            zeta_sd[f'zeta_{e}'] = np.std(zeta[f'zeta_{e}'], axis=(0, 1))
            zeta_range[f'zeta_{e}'] = np.vstack((zeta_mean[f'zeta_{e}'] - 1.96 * zeta_sd[f'zeta_{e}'],
                                                 zeta_mean[f'zeta_{e}'] + 1.96 * zeta_sd[f'zeta_{e}']))

            time_points = zeta_mean[f'zeta_{e}'] < threshold
            time_points = time_points[np.newaxis, np.newaxis, :]
            beta[f"beta_1_{e}"] = np.where(time_points, beta[f"beta_0_{e}"], beta[f"beta_1_{e}"])

            for s in range(2):
                # beta_mean[f'beta_{s}_{e}'] = np.mean(beta[f'beta_{s}_{e}'], axis=(0, 1))
                if s == 0:
                    beta_mean[f'beta_{s}_{e}'] = alpha_mean[f'alpha_{s}_{e}']
                elif s == 1:
                    beta_mean[f'beta_{s}_{e}'] = np.where(time_points, alpha_mean[f'alpha_{0}_{e}'],
                                                          zeta_mean[f'zeta_{e}'] * alpha_mean[f'alpha_{s}_{e}'] + \
                                                          (1 - zeta_mean[f'zeta_{e}']) * alpha_mean[f'alpha_{0}_{e}'])
                    beta_mean[f'beta_{s}_{e}'] = beta_mean[f'beta_{s}_{e}'].flatten()
                beta_sd[f'beta_{s}_{e}'] = np.std(beta[f'beta_{s}_{e}'], axis=(0, 1))
                beta_range[f'beta_{s}_{e}'] = np.vstack((beta_mean[f'beta_{s}_{e}'] - 1.96 * beta_sd[f'beta_{s}_{e}'],
                                                         beta_mean[f'beta_{s}_{e}'] + 1.96 * beta_sd[f'beta_{s}_{e}']))

        zeta_dfs = []
        for e in range(1, E + 1):
            zeta_df = pd.DataFrame({
                'Time': np.linspace(0, 1000, T),
                'Zeta_mean': zeta_mean[f'zeta_{e}'],
                'Zeta_min': zeta_range[f'zeta_{e}'][0],
                'Zeta_max': zeta_range[f'zeta_{e}'][1],
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
                'Channel': f'Channel {e}'
            })
            beta_dfs.append(beta_df)
            beta_df.to_csv(f'{R_path}/beta_{e}.csv', index=False)

    if method == 2:
        beta_mean = {}
        beta_sd = {}
        beta_range = {}
        for s in range(2):
            for e in range(1, E + 1):
                beta_mean[f'beta_{s}_{e}'] = np.mean(beta[f'beta_{s}_{e}'], axis=(0, 1))
                beta_sd[f'beta_{s}_{e}'] = np.std(beta[f'beta_{s}_{e}'], axis=(0, 1))
                beta_range[f'beta_{s}_{e}'] = np.vstack((beta_mean[f'beta_{s}_{e}'] - 1.96 * beta_sd[f'beta_{s}_{e}'],
                                                         beta_mean[f'beta_{s}_{e}'] + 1.96 * beta_sd[f'beta_{s}_{e}']))

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
                'Channel': f'Channel {e}'
            })
            beta_dfs.append(beta_df)
            beta_df.to_csv(f'{R_path}/beta_{e}.csv', index=False)

    Mu_mean = {}
    for s in range(2):
        Mu_mean[f'Mu_{s}'] = np.vstack([beta_mean[f'beta_{s}_{e}'] for e in range(1, E + 1)])

    C_t_mean = np.mean(C_t, axis=(0, 1))
    C_e_mean = np.mean(C_e, axis=(0, 1))

    train_acc, train_cum_score = predict(training, Mu_mean["Mu_1"], Mu_mean["Mu_0"], C_e_mean, C_t_mean, T, E)
    test_acc, test_cum_score = predict(testing, Mu_mean["Mu_1"], Mu_mean["Mu_0"], C_e_mean, C_t_mean, T, E)
    train_array = np.array(train_acc)
    test_array = np.array(test_acc)
    np.savetxt(f'{R_path}/train_acc.csv', train_array, delimiter=',')
    np.savetxt(f'{R_path}/test_acc.csv', test_array, delimiter=',')

    # visualize alpha and beta
    for e in range(1, E + 1):
        fig, ax = plt.subplots(1, 2, figsize=(24, 12))
        ax[0].plot(np.arange(T), alpha_mean[f'alpha_0_{e}'], label=f'Alpha_0', color='blue')
        ax[0].fill_between(np.arange(T), alpha_range[f'alpha_0_{e}'][0], alpha_range[f'alpha_0_{e}'][1], alpha=0.2,
                           color='blue')
        ax[0].plot(np.arange(T), alpha_mean[f'alpha_1_{e}'], label=f'Alpha_1', color='red')
        ax[0].fill_between(np.arange(T), alpha_range[f'alpha_1_{e}'][0], alpha_range[f'alpha_1_{e}'][1], alpha=0.2,
                           color='red')
        ax[0].legend()

        ax[1].plot(np.arange(T), beta_mean[f'beta_0_{e}'], label=f'Beta_0', color='blue')
        ax[1].fill_between(np.arange(T), beta_range[f'beta_0_{e}'][0], beta_range[f'beta_0_{e}'][1], alpha=0.2,
                           color='blue')
        ax[1].plot(np.arange(T), beta_mean[f'beta_1_{e}'], label=f'Beta_1', color='red')
        ax[1].fill_between(np.arange(T), beta_range[f'beta_1_{e}'][0], beta_range[f'beta_1_{e}'][1], alpha=0.2,
                           color='red')
        ax[1].legend()
        ax[0].set_title(f'Alpha for Channel {e}')
        ax[1].set_title(f'Beta for Channel {e}')
        plt.savefig(f'{plot_path}/Alpha and Beta for Channel {e}.png')
        plt.close()

    if method == 1:
        # visualize zeta
        for e in range(1, E + 1):
            plt.figure(figsize=(12, 6))
            plt.plot(np.arange(T), zeta_mean[f'zeta_{e}'])
            plt.fill_between(np.arange(T), zeta_range[f'zeta_{e}'][0], zeta_range[f'zeta_{e}'][1], alpha=0.2)
            plt.ylim(0, 1)
            plt.title(f'Zeta for Channel {e}')
            plt.savefig(f'{plot_path}/Zeta for Channel {e}.png')
            plt.close()

    # visualize train and test accuracy
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(np.unique(training['Sequence']), train_acc)
    ax[0].set_title('Train Accuracy')
    ax[0].set_xlabel('Number of Sequence')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim(0, 1)

    ax[1].plot(np.unique(testing['Sequence']), test_acc)
    ax[1].set_title('Test Accuracy')
    ax[1].set_xlabel('Number of Sequence')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_ylim(0, 1)
    plt.savefig(f'{plot_path}/Train and Test Accuracy.png')
    plt.close()
