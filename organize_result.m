% Script to organize hyperparameter search results
% Create output directory if it doesn't exist
outputDir = fullfile('Results', 'ds002778', 'HyperparameterSearchOrganized');
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% Source directory
sourceDir = fullfile('Results', 'ds002778', 'HyperparameterSearch');

% Step 1: Get all .mat files in the source directory
files = dir(fullfile(sourceDir, '*.mat'));
fileNames = {files.name};

% Step 2: Group files by model type
% Initialize containers for different model types
modelGroups = containers.Map;

% First pass: Group files by base model name
for i = 1:length(fileNames)
    fileName = fileNames{i};
    % Extract base name (remove _n.mat)
    baseName = regexp(fileName, '(.+)_\d+\.mat', 'tokens');
    if ~isempty(baseName)
        baseName = baseName{1}{1};
        if ~modelGroups.isKey(baseName)
            modelGroups(baseName) = {};
        end
        currentGroup = modelGroups(baseName);
        modelGroups(baseName) = [currentGroup, fileName];
    end
end

% Step 3: Process each model group
keys = modelGroups.keys;
for i = 1:length(keys)
    baseModelName = keys{i};
    modelFiles = modelGroups(baseModelName);
    
    % Special handling for RNN results
    if strcmp(baseModelName, 'rnnResults')
        % Load first file to get structure
        data = load(fullfile(sourceDir, modelFiles{1}));
        rnnData = data.rnnResults;
        
        % Initialize containers for different RNN types
        rnnTypes = ["gruLayer", "lstmLayer", "bigruLayerGraph", "bilstmLayer"];
        
        % Initialize empty structs for each RNN type
        gruLayer_results = struct([]);
        lstmLayer_results = struct([]);
        bigruLayerGraph_results = struct([]);
        bilstmLayer_results = struct([]);
        
        % Process all RNN files
        for fileIdx = 1:length(modelFiles)
            data = load(fullfile(sourceDir, modelFiles{fileIdx}));
            rnnResults = data.rnnResults;
            
            % Sort results by RNN type
            for j = 1:length(rnnResults)
                rnnType = rnnResults(j).parameters.typeOfUnit;
                % Clean the result structure
                cleanResult = struct('parameters', rnnResults(j).parameters, ...
                                    'performance', rnnResults(j).performance);
                if isfield(rnnResults(j), 'numberOfWeights')
                    cleanResult.numberOfWeights = rnnResults(j).numberOfWeights;
                end
                
                if strcmp(rnnType, 'gruLayer')
                    gruLayer_results = [gruLayer_results, cleanResult];
                elseif strcmp(rnnType, 'lstmLayer')
                    lstmLayer_results = [lstmLayer_results, cleanResult];
                elseif strcmp(rnnType, 'bigruLayerGraph')
                    bigruLayerGraph_results = [bigruLayerGraph_results, cleanResult];
                elseif strcmp(rnnType, 'bilstmLayer')
                    bilstmLayer_results = [bilstmLayer_results, cleanResult];
                end
            end
        end
        
        % Save separate files for each RNN type
        save(fullfile(outputDir, 'rnn_gru_results.mat'), 'gruLayer_results');
        save(fullfile(outputDir, 'rnn_lstm_results.mat'), 'lstmLayer_results');
        save(fullfile(outputDir, 'rnn_bigru_results.mat'), 'bigruLayerGraph_results');
        save(fullfile(outputDir, 'rnn_bilstm_results.mat'), 'bilstmLayer_results');
        
    % Handle CNNRNN and TCNRNN results
    elseif ~isempty(strfind(baseModelName, 'rnn'))  % Using strfind instead of contains for backwards compatibility
        % Load and combine all channel versions
        allResults = struct([]);
        for fileIdx = 1:length(modelFiles)
            data = load(fullfile(sourceDir, modelFiles{fileIdx}));
            fieldName = baseModelName;
            allResults = [allResults, data.(fieldName)];
        end
        
        % Split by RNN unit type
        bigruResults = struct([]);
        bilstmResults = struct([]);
        
        for j = 1:length(allResults)
            % Clean the result structure
            cleanResult = struct('parameters', allResults(j).parameters, ...
                                'performance', allResults(j).performance);
            if isfield(allResults(j), 'numberOfWeights')
                cleanResult.numberOfWeights = allResults(j).numberOfWeights;
            end
            
            if strcmp(allResults(j).parameters.typeOfUnit, 'bigruLayerGraph')
                bigruResults = [bigruResults, cleanResult];
            elseif strcmp(allResults(j).parameters.typeOfUnit, 'bilstmLayer')
                bilstmResults = [bilstmResults, cleanResult];
            end
        end
        
        % Determine model type and connection type
        isDW = ~isempty(strfind(baseModelName, 'DW'));
        isTCN = ~isempty(strfind(baseModelName, 'tcn'));
        
        % Create appropriate filenames
        if isTCN
            prefix = 'tcnrnn';
        else
            prefix = 'cnnrnn';
        end
        
        if isDW
            connType = '_dw';
        else
            connType = '';
        end
        
        % Save separate files for BiGRU and BiLSTM
        save(fullfile(outputDir, [prefix, connType, '_bigru_results.mat']), 'bigruResults');
        save(fullfile(outputDir, [prefix, connType, '_bilstm_results.mat']), 'bilstmResults');
        
    % Handle regular CNN and TCN results
    else
        % Load and combine all channel versions
        allResults = struct([]);
        for fileIdx = 1:length(modelFiles)
            data = load(fullfile(sourceDir, modelFiles{fileIdx}));
            fieldName = baseModelName;
            allResults = [allResults, data.(fieldName)];
        end
        
        % Determine model type and connection type
        isDW = ~isempty(strfind(baseModelName, 'DW'));
        isTCN = ~isempty(strfind(baseModelName, 'tcn'));
        isStandard = ~isempty(strfind(baseModelName, 'Standard'));
        
        % Create appropriate filename
        if isTCN
            if isDW
                outFileName = 'tcn_dw_results.mat';
            elseif isStandard
                outFileName = 'tcn_standard_results.mat';
            else
                outFileName = 'tcn_results.mat';
            end
        else % CNN
            if isDW
                outFileName = 'cnn_dw_results.mat';
            else
                outFileName = 'cnn_results.mat';
            end
        end
        
        % Save combined results
        results = allResults;
        save(fullfile(outputDir, outFileName), 'results');
    end
end

% Display completion message
fprintf('Results organization complete. Files saved in: %s\n', outputDir);