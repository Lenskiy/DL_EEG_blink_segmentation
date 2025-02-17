%% Analysis and LaTeX Table Generation Script
% Load and analyze hyperparameter search results, generate IEEE style LaTeX tables

%% Setups
inputDir = fullfile('Results', 'ds002778', 'HyperparameterSearchOrganized');
outputDir = fullfile('Results', 'ds002778', 'HyperparameterSearchLatex');

phase = '_hyperparameter_search';
if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

%% Helper functions
function str = formatValue(value, std)
    % Format value with standard deviation as value±std
    str = sprintf('%.3f$\\pm$%.3f', value, std);
end

function str = formatValueBold(value, std)
    % Format value with standard deviation in bold
    str = sprintf('\\textbf{%.3f$\\pm$%.3f}', value, std);
end

function [bestValue, bestIndex] = findBestInColumn(data, colIdx)
    % Find the best (maximum) value in a specific column
    values = cellfun(@(x) str2double(regexp(x, '\d+\.\d+', 'match', 'once')), data(:,colIdx));
    [bestValue, bestIndex] = max(values);
end

function writeLatexTable(filename, caption, header, data, label)
    fid = fopen(filename, 'w');
    
    fprintf(fid, '\\begin{table*}[!t]\n');
    fprintf(fid, '\\renewcommand{\\arraystretch}{1.3}\n');
    fprintf(fid, '\\caption{%s}\n', caption);
    fprintf(fid, '\\label{%s}\n', label);
    fprintf(fid, '\\centering\n');
    
    % Calculate column format based on number of columns
    colFormat = repmat('c', 1, size(data, 2));
    fprintf(fid, '\\begin{tabular}{|%s|}\n', colFormat);
    fprintf(fid, '\\hline\n');
    
    % Write headers
    fprintf(fid, '%s \\\\ \\hline\\hline\n', strjoin(header, ' & '));
    
    % Write data rows
    for i = 1:size(data, 1)
        fprintf(fid, '%s \\\\ \\hline\n', strjoin(data(i,:), ' & '));
    end
    
    fprintf(fid, '\\end{tabular}\n');
    fprintf(fid, '\\end{table*}\n');
    
    fclose(fid);
end

function [bestResult, bestIdx] = findBestResult(results, metric)
    % Find best result based on specified metric
    allMetrics = zeros(1, length(results));
    for i = 1:length(results)
        allMetrics(i) = results(i).performance.(metric)(1);
    end
    [~, bestIdx] = max(allMetrics);
    bestResult = results(bestIdx);
end

%% Load all result files
files = dir(fullfile(inputDir, '*.mat'));
results = struct();

for i = 1:length(files)
    [~, name, ~] = fileparts(files(i).name);
    data = load(fullfile(inputDir, files(i).name));
    fieldNames = fieldnames(data);
    results.(name) = data.(fieldNames{1});
end

%% 1. Best Performance Analysis Table
modelNames = {
    'cnn_results', 'cnn_dw_results', ...
    'tcn_results', 'tcn_standard_results', ...
    'rnn_lstm_results', 'rnn_gru_results', 'rnn_bilstm_results', 'rnn_bigru_results', ...
    'cnnrnn_bigru_results', 'cnnrnn_bilstm_results', ...
    'cnnrnn_dw_bigru_results', 'cnnrnn_dw_bilstm_results', ...
    'tcnrnn_bigru_results', 'tcnrnn_bilstm_results', ...
    'tcnrnn_dw_bigru_results', 'tcnrnn_dw_bilstm_results'
};

displayNames = {
    'CNN', 'CNN-DW', ...
    'TCN', 'TCN-Standard', ...
    'RNN-LSTM', 'RNN-GRU', 'RNN-BiLSTM', 'RNN-BiGRU', ...
    'CNNRNN-BiGRU', 'CNNRNN-BiLSTM', ...
    'CNNRNN-DW-BiGRU', 'CNNRNN-DW-BiLSTM', ...
    'TCNRNN-BiGRU', 'TCNRNN-BiLSTM', ...
    'TCNRNN-DW-BiGRU', 'TCNRNN-DW-BiLSTM'
};

% Initialize performance table
perfTable = cell(length(modelNames), 7);
bestF1 = 0;
bestF1v = 0;
bestAcc = 0;
bestBDR = 0;
bestBDRv = 0;

% Fill performance table
for i = 1:length(modelNames)
    if isfield(results, modelNames{i})
        [bestResult, ~] = findBestResult(results.(modelNames{i}), 'f1Score');
        
        f1 = bestResult.performance.f1Score;
        f1v = bestResult.performance.f1Score_v;
        acc = bestResult.performance.macroAccuracy;
        bdr = bestResult.performance.blinkDetectionRate;
        bdrv = bestResult.performance.blinkDetectionRate_v;
        
        % Update best values
        bestF1 = max(bestF1, f1(1));
        bestF1v = max(bestF1v, f1v(1));
        bestAcc = max(bestAcc, acc(1));
        bestBDR = max(bestBDR, bdr(1));
        bestBDRv = max(bestBDRv, bdrv(1));
        
        % Store formatted values
        perfTable(i,:) = {
            displayNames{i}, ...
            formatValue(f1(1), f1(2)), ...
            formatValue(f1v(1), f1v(2)), ...
            formatValue(acc(1), acc(2)), ...
            formatValue(bdr(1), bdr(2)), ...
            formatValue(bdrv(1), bdrv(2)), ...
            num2str(bestResult.numberOfWeights)
        };
    end
end

% Bold the best values
for i = 1:size(perfTable,1)
    % F1-Score
    if ~isempty(perfTable{i,2})
        values = regexp(perfTable{i,2}, '\d+\.\d+', 'match');
        if length(values) >= 2 && str2double(values{1}) == bestF1
            perfTable{i,2} = formatValueBold(bestF1, str2double(values{2}));
        end
    end
    
    % F1-Score (v)
    if ~isempty(perfTable{i,3})
        values = regexp(perfTable{i,3}, '\d+\.\d+', 'match');
        if length(values) >= 2 && str2double(values{1}) == bestF1v
            perfTable{i,3} = formatValueBold(bestF1v, str2double(values{2}));
        end
    end
    
    % Accuracy
    if ~isempty(perfTable{i,4})
        values = regexp(perfTable{i,4}, '\d+\.\d+', 'match');
        if length(values) >= 2 && str2double(values{1}) == bestAcc
            perfTable{i,4} = formatValueBold(bestAcc, str2double(values{2}));
        end
    end
    
    % BDR
    if ~isempty(perfTable{i,5})
        values = regexp(perfTable{i,5}, '\d+\.\d+', 'match');
        if length(values) >= 2 && str2double(values{1}) == bestBDR
            perfTable{i,5} = formatValueBold(bestBDR, str2double(values{2}));
        end
    end
    
    % BDR (v)
    if ~isempty(perfTable{i,6})
        values = regexp(perfTable{i,6}, '\d+\.\d+', 'match');
        if length(values) >= 2 && str2double(values{1}) == bestBDRv
            perfTable{i,6} = formatValueBold(bestBDRv, str2double(values{2}));
        end
    end
end

% Write performance table
header = {'Model', 'F1-Score', 'F1-Score (v)', 'Accuracy', 'BDR', 'BDR (v)', 'Params'};
writeLatexTable(fullfile(outputDir, ['performance_comparison' phase '.tex']), ...
    'Performance Comparison of Different Models', ...
    header, perfTable, 'tab:perf_comparison');

%% 2. Best Hyperparameters Table
hyperTable = cell(length(modelNames), 6);
for i = 1:length(modelNames)
    if isfield(results, modelNames{i})
        [bestResult, ~] = findBestResult(results.(modelNames{i}), 'f1Score');
        params = bestResult.parameters;
        
        % Format hyperparameters based on model type
        if contains(modelNames{i}, 'cnn') || contains(modelNames{i}, 'tcn')
            hyperTable(i,:) = {
                displayNames{i}, ...
                num2str(params.numBlocks), ...
                num2str(params.filterSize), ...
                num2str(params.numFilters), ...
                '-', ...
                '-'
            };
        elseif contains(modelNames{i}, 'rnn')
            if isfield(params, 'numRBlocks')
                hyperTable(i,:) = {
                    displayNames{i}, ...
                    '-', ...
                    '-', ...
                    '-', ...
                    num2str(params.numRBlocks), ...
                    num2str(params.numUnits)
                };
            end
        end
    end
end

header = {'Model', 'Blocks', 'Filter Size', 'Filters', 'RNN Blocks', 'Units'};
writeLatexTable(fullfile(outputDir, ['hyperparameters' phase '.tex']), ...
    'Best Hyperparameters for Each Model', ...
    header, hyperTable, 'tab:hyperparameters');

%% 3. Class-wise Performance Analysis % No. Not a reasonable compare.
% classTable = cell(length(modelNames), 5);
% for i = 1:length(modelNames)
%     if isfield(results, modelNames{i})
%         [bestResult, ~] = findBestResult(results.(modelNames{i}), 'f1Score');
%         perf = bestResult.performance;
        
%         classTable(i,:) = {
%             displayNames{i}, ...
%             formatValue(perf.blink_accuracy(1), perf.blink_accuracy(2)), ...
%             formatValue(perf.n_a_accuracy(1), perf.n_a_accuracy(2)), ...
%             formatValue(perf.muscle_artifact_accuracy(1), perf.muscle_artifact_accuracy(2)), ...
%             formatValue(perf.macroAccuracy(1), perf.macroAccuracy(2))
%         };
%     end
% end

% header = {'Model', 'Blink Acc.', 'N/A Acc.', 'Artifact Acc.', 'Macro Acc.'};
% writeLatexTable(fullfile(outputDir, ['class_performance' phase '.tex']), ...
%     'Class-wise Performance Analysis', ...
%     header, classTable, 'tab:class_perf');

%% 4. Voting Impact Analysis
votingTable = cell(length(modelNames), 7);
bestImprovementF1 = 0;
bestImprovementBDR = 0;

for i = 1:length(modelNames)
    if isfield(results, modelNames{i})
        [bestResult, ~] = findBestResult(results.(modelNames{i}), 'f1Score');
        perf = bestResult.performance;
        
        % Calculate improvements
        f1_improvement = ((perf.f1Score_v(1) - perf.f1Score(1)) / perf.f1Score(1)) * 100;
        bdr_improvement = ((perf.blinkDetectionRate_v(1) - perf.blinkDetectionRate(1)) / perf.blinkDetectionRate(1)) * 100;
        
        % Update best improvements
        bestImprovementF1 = max(bestImprovementF1, f1_improvement);
        bestImprovementBDR = max(bestImprovementBDR, bdr_improvement);
        
        votingTable(i,:) = {
            displayNames{i}, ...
            formatValue(perf.f1Score(1), perf.f1Score(2)), ...
            formatValue(perf.f1Score_v(1), perf.f1Score_v(2)), ...
            sprintf('%.2f\\%%', f1_improvement), ...
            formatValue(perf.blinkDetectionRate(1), perf.blinkDetectionRate(2)), ...
            formatValue(perf.blinkDetectionRate_v(1), perf.blinkDetectionRate_v(2)), ...
            sprintf('%.2f\\%%', bdr_improvement)
        };
    end
end

% Bold the best improvements
for i = 1:size(votingTable,1)
    if ~isempty(votingTable{i,4})
        improvement = str2double(regexp(votingTable{i,4}, '-?\d+\.\d+', 'match'));
        if improvement == bestImprovementF1
            votingTable{i,4} = sprintf('\\textbf{%.2f\\%%}', improvement);
        end
    end
    if ~isempty(votingTable{i,7})
        improvement = str2double(regexp(votingTable{i,7}, '-?\d+\.\d+', 'match'));
        if improvement == bestImprovementBDR
            votingTable{i,7} = sprintf('\\textbf{%.2f\\%%}', improvement);
        end
    end
end

% Write voting impact table
header = {'Model', 'F1-Score', 'F1-Score (v)', 'F1 Imp.', 'BDR', 'BDR (v)', 'BDR Imp.'};
writeLatexTable(fullfile(outputDir, ['voting_impact' phase '.tex']), ...
    'Impact of Voting Mechanism on Model Performance', ...
    header, votingTable, 'tab:voting_impact');