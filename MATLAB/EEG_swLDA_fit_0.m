clear

persons = {"K178"};

for i = 1:length(persons)
    person = persons{i};  
    file_path = fullfile("E:\MSPH\EEG methodology\Advanced EEG Code\Code for GitHub\SMGP_BCI_EEG\EEG_multi\", person);
    
    mat_file = fullfile(file_path, person + "_TRN_xDAWN.mat");

    data = load(mat_file);

    X_train = data.X;
    y_train = data.Y;

    X_train = double(X_train);
    y_train = double(y_train);

    [b, ~, ~, inmodel, ~] = trainSWLDAmatlab(X_train, y_train, 15);
    b(inmodel.' == 0, :) = 0;
    Score = X_train * b;
    Score_1 = Score(y_train == 1, :);
    Score_0 = Score(y_train ~= 1, :);
    Mean_1 = mean(Score_1);
    Mean_0 = mean(Score_0);
    Std = sqrt(var(Score));

    save_filename = sprintf('%s_train_data_swLDA.mat', person);
    save_fullpath = fullfile(file_path, save_filename);

    % save files
    save(save_fullpath, 'b', 'Mean_1', 'Mean_0', 'Std');

end
    
