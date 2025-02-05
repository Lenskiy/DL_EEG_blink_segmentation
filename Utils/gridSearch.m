function results = gridSearch(model, trainSetX, trainSetY, trainXPID, testSetX, testSetY, testXPID , parameters, options, numTrials, stepLenght, iterateOver)
    rng(2,"twister");
    listOfParamters = cartesianProduct(parameters, iterateOver);

    results(length(listOfParamters)) = struct('parameters', [], 'performance', [], 'numberOfWeights', []);

    po = [];    % variable to parallel pool
    for k = length(listOfParamters):-1:1       % Filter size search
        dlnet = model(listOfParamters(k));
        results(k) =  struct('parameters', listOfParamters(k), 'performance', [], 'numberOfWeights', []);
        performance = zeros(1,numTrials);
        tic
        if(options.ExecutionEnvironment == "gpu")
            numberProcesses = 1;
        else
            numberProcesses = min(numTrials, feature('numcores'));
        end
        try % don't stop and restart the pool, instead check if it's not runing then start 
            if isempty(po) && 1 < numTrials
                delete(gcp('nocreate'));
                po = parpool(numberProcesses);
            end
            
            if(numTrials == 1)
                 f = @() trainAndTestNetworks(trainSetX, trainSetY, trainXPID, testSetX, testSetY, testXPID, dlnet, stepLenght, options);
            else
                f(1:numTrials) = parallel.FevalFuture;
                for l = 1:numTrials
                    f(l) = parfeval(po, @trainAndTestNetworks, 3, trainSetX, trainSetY, trainXPID, testSetX, testSetY, testXPID, dlnet, stepLenght, options);
                end
                wait(f);
            end
        catch EM
            delete(gcp('nocreate'));
            disp(EM)
        end
        
        
        if(numTrials == 1)
            [~, performance, numberOfWeights] = f();        
        else
            for l = 1:numTrials
                [~, performance_temp, numberOfWeights]= f(l).fetchOutputs;

                fnames = fieldnames(performance_temp);
                if l == 1
                    performance = performance_temp;
                else
                    for m = 1:length(fnames)
                        performance.(string(fnames(m))) = [performance.(string(fnames(m))), performance_temp.(string(fnames(m)))];
                    end
                end
            end
        end
        t = toc;
        performanceStats = struct();
        fnames = fieldnames(performance);
        for m = 1:length(fnames)
            performanceStats.(string(fnames(m))) = [mean(performance.(string(fnames(m)))), std(performance.(string(fnames(m))))];
        end

        results(k).('performance')     = performanceStats;
        results(k).('numberOfWeights') = numberOfWeights;

        save(fullfile("Results", string(char(model)) + "_temp.mat"), 'results');
        % dont print parallel column
        tmpTable = struct2table(results(k).parameters);
        disp([table(k, numberOfWeights, t/60, VariableNames = ["k", "numberOfWeights", "time"]),...
                        tmpTable(:, iterateOver), struct2table(performanceStats)]);
        if(gpuDeviceCount ~= 0)
            gpuDevice(1);
        end
    end
end



function [net, performance_report, numberOfWeights] = trainAndTestNetworks(trainSetX, trainSetY, trainXPID, testSetX, testSetY, testXPID, dlnet, stepLenght, options)
    % Format training data
    [dlX, dlY] = formatData(trainSetX, trainSetY);
    
    % Format validation data
    if ~isempty(options.ValidationData)
        valData = options.ValidationData;
        [dlValX, dlValY] = formatData(valData{1}, valData{2});
    end

    % Add validation data if it exists
    if exist('dlValX', 'var') && exist('dlValY', 'var')
        options.ValidationData = {dlValX, dlValY};
    end

    % Rest of the code remains the same...
    [net, ~] = trainnet(dlX, dlY, dlnet, @sequenceCrossEntropyLoss, options);
    
    % Format test data and predict
    [dlTestX, ~] = formatData(testSetX, testSetY);
    rawPredictions = predict(net, dlTestX);
    
    % Convert predictions to cell array of categorical arrays
    YPredicted = convertPredictionsToCategorical(rawPredictions, testSetY);
    
    % Apply voting mechanism
    YPredictedReArranged = mergeSegmentsByVoting2(YPredicted, testXPID, stepLenght);
 
    % Compute performance metrics using rearranged predictions
    performanceReport = computeModelPerformance(YPredicted, testSetY, trainSetX);
    performanceReport_v = computeModelPerformance(YPredictedReArranged, testSetY, trainSetX);

    labels = string(categories(testSetY{1}));

    performance_report = struct(labels(1).replace("/","_").replace("-","_") + "_accuracy", performanceReport.perClassAccuracy(1), labels(1).replace("/","_").replace("-","_") + "_accuracy_v", performanceReport_v.perClassAccuracy(1),...
                                labels(2).replace("/","_").replace("-","_") + "_accuracy", performanceReport.perClassAccuracy(2), labels(2).replace("/","_").replace("-","_") + "_accuracy_v", performanceReport_v.perClassAccuracy(2),...
                                labels(3).replace("/","_").replace("-","_") + "_accuracy", performanceReport.perClassAccuracy(3), labels(3).replace("/","_").replace("-","_") + "_accuracy_v", performanceReport_v.perClassAccuracy(3),...
                                "macroAccuracy", performanceReport.macroAccuracy,   "macroAccuracy_v", performanceReport_v.macroAccuracy,...
                                "f1Score",       performanceReport.f1Score,         "f1Score_v", performanceReport_v.f1Score,...
                                "precision",     performanceReport.precision,       "precision_v", performanceReport_v.precision,...
                                "recall",        performanceReport.recall,          "recall_v", performanceReport_v.recall,...
                                "blinkDetectionRate", performanceReport.blinkDetectionRate,  "blinkDetectionRate_v", performanceReport_v.blinkDetectionRate);

    info = analyzeNetwork(net, Plots="none");
    numberOfWeights = info.TotalLearnables;
end


function categoricalPredictions = convertPredictionsToCategorical(predictions, originalY)
    cats = categories(originalY{1});
    numSequences = size(predictions, 2);
    
    categoricalPredictions = cell(1, numSequences);
    for i = 1:numSequences
        seqPreds = squeeze(predictions(:, i, :));
        [~, maxIdx] = max(seqPreds, [], 1);
        categoricalPredictions{i} = categorical(cats(maxIdx)', cats);
    end
end

% Helper function to format data consistently
function [dlX, dlY] = formatData(X, Y)
    numSequences = numel(X);
    seqLength = size(X{1}, 2);
    numChannels = size(X{1}, 1);
    
    % Format input data
    formattedX = zeros(numChannels, numSequences, seqLength);
    for i = 1:numSequences
        formattedX(:,i,:) = X{i};
    end
    
    % Format labels
    numClasses = numel(categories(Y{1}));
    formattedY = zeros(numClasses, numSequences, seqLength);
    for i = 1:numSequences
        catArray = Y{i};
        oneHot = onehotencode(catArray', 2);
        formattedY(:,i,:) = permute(oneHot, [2 3 1]);
    end
    
    % Convert to dlarrays
    dlX = dlarray(formattedX, 'CBT');
    dlY = dlarray(formattedY, 'CBT');
end

function loss = sequenceCrossEntropyLoss(predictions, targets)
    % predictions are already probabilities from softmax layer
    loss = -sum(targets .* log(predictions + eps), 'all');
    numElements = size(targets, 2) * size(targets, 3);
    loss = loss / numElements;
end
%% OLD functions
% function [net, performance_, numberOfWeights]   = trainAndTestNetworks(trainSetX, trainSetY, trainXPID, testSetX, testSetY, testXPID, net, stepLenght, options)
%     [net, ~]        = trainNetwork(trainSetX, trainSetY, net, options);
%     YPredicted      = classify(net, testSetX)';
%     numberOfWeights = computeNumberOfWeights(net);
% 
%     % Use overlapping segments to vote for the final label per sample
%     YPredictedReArranged = mergeSegmentsByVoting2(YPredicted, testXPID, stepLenght);
% 
%     %accuracy        = 100*mean(cellfun(@(y1, y2) sum(y1 == y2)/length(y1), YPredictedReArranged, testSetY));
%     %accuracy        = 100*mean(cellfun(@(y1, y2) sum(y1 == y2)/length(y1), YPredicted, testSetY));
% 
%     accuracy = performanceMetrics(YPredictedReArranged, YPredicted, "accuracy", trainSetX); 
%     f1Score = performanceMetrics(YPredictedReArranged, YPredicted, "f1Score", trainSetX);
%     %f2Score = performanceMetrics(YPredictedReArranged, YPredicted, "f2Score", trainSetX);
%     blinkInRange = performanceMetrics(YPredictedReArranged, YPredicted, "blinkInRange", trainSetX);
%     blinkFound = performanceMetrics(YPredictedReArranged, YPredicted, "blinkFound", trainSetX);
%     f1Macro = performanceMetrics(YPredictedReArranged, YPredicted, "macroF1Score", trainSetX);
% 
%     performance_ = struct("accuracy", accuracy, "f1Score", f1Score, "blinkInRange", blinkInRange, "blinkFound", blinkFound, "f1Macro", f1Macro);
% end

% function numberOfWeights = computeNumberOfWeights(lgraph)
%     numberOfWeights = 0;
%     for k = 1:numel(lgraph.Layers)
%         if isprop(lgraph.Layers(k), "Weights")
%             numberOfWeights = numberOfWeights + numel(lgraph.Layers(k).Weights);
%         end
%         if isprop(lgraph.Layers(k), "Bias")
%             numberOfWeights = numberOfWeights + numel(lgraph.Layers(k).Bias);
%         end
%         if isprop(lgraph.Layers(k), "InputWeights")
%             numberOfWeights = numberOfWeights + numel(lgraph.Layers(k).InputWeights);
%         end
%         if isprop(lgraph.Layers(k), "RecurrentWeights")
%             numberOfWeights = numberOfWeights + numel(lgraph.Layers(k).RecurrentWeights);
%         end
%         if isprop(lgraph.Layers(k), "PeepholeWeights")
%             numberOfWeights = numberOfWeights + numel(lgraph.Layers(k).PeepholeWeights);
%         end
%         if(isprop(lgraph.Layers(k), "Network"))
%             layers = lgraph.Layers(k).Network.Learnables.Value;
%             numberOfWeights = numberOfWeights + sum(cellfun(@(x)numel(x), layers));
%         end
%     end
% end