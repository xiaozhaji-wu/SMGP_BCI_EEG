import numpy as np
import pandas as pd
import scipy.io
from multi_visual_Func import predict_swLDA
from SIM_generate import L, I

replications = 100

train_acc_list = []
test_acc_list = []
for rep in range(replications):
    # read data from matlab data
    swLDA_output = scipy.io.loadmat(f'SIM_multi/replication_{rep}/train_data_L_{L}_I_{I}_{rep}.mat')

    b_inmodel = swLDA_output['b']
    mean_1 = swLDA_output['Mean_1'][0, 0]
    mean_0 = swLDA_output['Mean_0'][0, 0]
    std = swLDA_output['Std'][0, 0]

    train = pd.read_csv(f'SIM_multi/replication_{rep}/train_data_L_{L}_I_{I}_{rep}.csv')
    test = pd.read_csv(f'SIM_multi/replication_{rep}/test_data_L_{L}_I_{I}_{rep}.csv')

    train_acc, _ = predict_swLDA(train, b_inmodel, mean_1, mean_0, std, 30, 2)
    test_acc, _ = predict_swLDA(test, b_inmodel, mean_1, mean_0, std, 30, 2)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    np.savetxt(f'E:/MSPH/EEG methodology/Advanced EEG Code/SIM_multi/replication_{rep}/R_plots/L_{L}_I_{I}/swLDA_train_accuracy.csv', train_acc, delimiter=',')
    np.savetxt(f'E:/MSPH/EEG methodology/Advanced EEG Code/SIM_multi/replication_{rep}/R_plots/L_{L}_I_{I}/swLDA_test_accuracy.csv', test_acc, delimiter=',')

train_acc_array = np.array(train_acc_list)
test_acc_array = np.array(test_acc_list)

train_acc_mean = np.mean(train_acc_array, axis=0)
test_acc_mean = np.mean(test_acc_array, axis=0)

np.savetxt(f'E:/MSPH/EEG methodology/Advanced EEG Code/SIM_multi/R_plots/L_{L}_I_{I}/swLDA_train_accuracy.csv', train_acc_mean, delimiter=',')
np.savetxt(f'E:/MSPH/EEG methodology/Advanced EEG Code/SIM_multi/R_plots/L_{L}_I_{I}/swLDA_test_accuracy.csv', test_acc_mean, delimiter=',')
