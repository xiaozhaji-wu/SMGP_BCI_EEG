clear

persons = {"K106", "K107", "K108", "K111", "K112", "K113", "K114", "K118", ...
           "K121", "K123", "K143", "K145", "K146", "K147", "K151", "K160", ...
           "K171", "K172", "K177", "K178", "K183", "K190", "K191", "K159", ...
           "K185", "K184", "K154", "K166"};

for i = 1:length(persons)
    person = persons{i};  
    file_path = fullfile("E:\MSPH\EEG methodology\Advanced EEG Code\EEG_multi", person);
    
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
    
