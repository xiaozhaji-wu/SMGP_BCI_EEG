clear

L = 20;
I = 10;

for rep = 0:99

file_name = sprintf('E:/MSPH/EEG methodology/Advanced EEG Code/SIM_multi/replication_%d/train_data_L_%d_I_%d_%d.csv', rep, L, I, rep);
train_df = readtable(file_name);


[X_train, y_train] = prepareDataForSWLDA(train_df, 30);


[b, ~, ~, inmodel, ~] = trainSWLDAmatlab(X_train, y_train, 18);

b(inmodel.' == 0, :) = 0;
Score = X_train * b;
Score_1 = Score(y_train == 1, :);
Score_0 = Score(y_train ~= 1, :);
Mean_1 = mean(Score_1);
Mean_0 = mean(Score_0);
Std = sqrt(var(Score));

% specify the save path and file name
save_path = sprintf('E:/MSPH/EEG methodology/Advanced EEG Code/SIM_multi/replication_%d/', rep);
save_filename = sprintf('train_data_L_%d_I_%d_%d.mat', L, I, rep);
save_fullpath = fullfile(save_path, save_filename);

% save mat file
save(save_fullpath, 'b', 'Mean_1', 'Mean_0', 'Std');

end
    
