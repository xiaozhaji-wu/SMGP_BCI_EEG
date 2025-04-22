import numpy as np
import scipy.io as sio
from EEG_visual_Func import predict_swLDA

persons = ["K106", "K107", "K113", "K114",
           "K145", "K151", "K154", "K159",
           "K160", "K172", "K178", "K183",
           "K184", "K185", "K190", "K191"]

train_acc_list = []
test_acc_list = []
for person in persons:
    # read data from matlab data
    swLDA_output = sio.loadmat(f'EEG_multi/{person}/{person}_train_data_swLDA.mat')

    b_inmodel = swLDA_output['b']
    mean_1 = swLDA_output['Mean_1'][0, 0]
    mean_0 = swLDA_output['Mean_0'][0, 0]
    std = swLDA_output['Std'][0, 0]

    train = sio.loadmat(f'EEG_multi/{person}/{person}_TRN_xDAWN.mat')
    test = sio.loadmat(f'EEG_multi/{person}/{person}_FRT_xDAWN.mat')

    train_acc, _ = predict_swLDA(train, b_inmodel, mean_1, mean_0, std, 25, 2)
    test_acc, _ = predict_swLDA(test, b_inmodel, mean_1, mean_0, std, 25, 2)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    np.savetxt(f'E:/MSPH/EEG methodology/Advanced EEG Code/EEG_multi/{person}/R_plots/swLDA_train_accuracy.csv', train_acc, delimiter=',')
    np.savetxt(f'E:/MSPH/EEG methodology/Advanced EEG Code/EEG_multi/{person}/R_plots/swLDA_test_accuracy.csv', test_acc, delimiter=',')



