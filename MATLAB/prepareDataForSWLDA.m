function [X_data, y_data] = prepareDataForSWLDA(df, T)
    % get unique sequences and letters
    sequences = unique(df.Sequence);
    letters = unique(df.Letter);

    
    % initialize the data used to store features and labels
    X_data = [];
    y_data = [];
    
    for i = 1:length(letters)
        letter = letters(i);
        letter_data = [];
        letter_label = [];
        
        for j = 1:length(sequences)
            seq = sequences(j);
            
            % extract the data corresponding to the current sequence and letter
            seq_letter_data = df(df.Sequence == seq & df.Letter == letter, :);
            
            % extract the first T column of features for two channels
            X_channel_1 = seq_letter_data{seq_letter_data.Channel == 1, 1:T};
            X_channel_2 = seq_letter_data{seq_letter_data.Channel == 2, 1:T};

            
            % combining the features of two channels into one feature matrix
            X_combined = [X_channel_1, X_channel_2];
            
            % extract labels
            y_combined = seq_letter_data{seq_letter_data.Channel == 1, 'Y'};
            
            % sorting data in flash order
            flash_order = seq_letter_data{seq_letter_data.Channel == 1, 'Flash'};
            [~, sorted_indices] = sort(flash_order);
            X_sorted = X_combined(sorted_indices, :);
            y_sorted = y_combined(sorted_indices);
            
            letter_data = [letter_data; X_sorted];
            letter_label = [letter_label; y_sorted];
        end
        
        X_data = [X_data; letter_data];
        y_data = [y_data; letter_label];
    end
    

