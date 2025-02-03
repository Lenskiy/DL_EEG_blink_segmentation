function performanceReport = computeModelPerformance(predicted, actual, signal)
    % COMPUTEMODELPERFORMANCE Computes comprehensive performance metrics
    %
    % Inputs:
    %   predicted - Cell array of categorical predictions
    %   actual    - Cell array of categorical ground truth
    %   signal    - Cell array of original signals (for peak detection)
    %
    % Returns:
    %   performanceReport - Struct containing all performance metrics

    % Initialize performance report
    performanceReport = struct();
    
    % 1. Compute macro-accuracy and per-class accuracies
    [macroAcc, perClassAcc] = computeDetailedAccuracy(predicted, actual);
    performanceReport.macroAccuracy = macroAcc;
    performanceReport.perClassAccuracy = perClassAcc;
    
    % 2. Compute F1-score
    [f1, precision, recall] = computeF1Metrics(predicted, actual);
    performanceReport.f1Score = f1;
    performanceReport.precision = precision;
    performanceReport.recall = recall;
    
    % 3. Compute blink detection rate
    blinkRate = computeBlinkDetectionRate(predicted, actual, signal);
    performanceReport.blinkDetectionRate = blinkRate;
end

function [macroAccuracy, perClassAccuracy] = computeDetailedAccuracy(predicted, actual)
    % Compute per-class and macro accuracies
    classes = categories(actual{1});
    perClassAccuracy = zeros(length(classes), 1);
    
    accuracySum = 0;
    for i = 1:length(classes)
        className = classes{i};
        classTotal = sum(cellfun(@(a) sum(a == className), actual));
        classCorrect = sum(cellfun(@(p,a) sum((p == className) & (a == className)), predicted, actual));
        
        if classTotal > 0
            classAcc = classCorrect/classTotal;
        else
            classAcc = 0;
        end
        
        perClassAccuracy(i) = classAcc;
        accuracySum = accuracySum + classAcc;
    end
    
    macroAccuracy = accuracySum / length(classes);
end

function [f1, precision, recall] = computeF1Metrics(predicted, actual)
    % Compute F1 score and its components
    blinkClass = 'blink';
    
    tp = sum(cellfun(@(p,a) sum((p == blinkClass) & (a == blinkClass)), predicted, actual));
    fp = sum(cellfun(@(p,a) sum((p == blinkClass) & (a ~= blinkClass)), predicted, actual));
    fn = sum(cellfun(@(p,a) sum((p ~= blinkClass) & (a == blinkClass)), predicted, actual));
    
    if tp + fp > 0
        precision = tp / (tp + fp);
    else
        precision = 0;
    end
    
    if tp + fn > 0
        recall = tp / (tp + fn);
    else
        recall = 0;
    end
    
    if precision + recall > 0
        f1 = 2 * (precision * recall) / (precision + recall);
    else
        f1 = 0;
    end
end

function blinkRate = computeBlinkDetectionRate(predicted, actual, signal)
    % Compute blink detection rate (blinkInRange)
    correctBlinks = 0;
    totalActualBlinks = 0;
    
    for i = 1:length(predicted)
        [correct, total] = countBlinkDetections(predicted{i}, actual{i}, signal{i});
        correctBlinks = correctBlinks + correct;
        totalActualBlinks = totalActualBlinks + total;
    end
    
    if totalActualBlinks > 0
        blinkRate = correctBlinks / totalActualBlinks;
    else
        blinkRate = 0;
    end
end

function [correct, total] = countBlinkDetections(predicted, actual, signal)
    % Count correct blink detections
    correct = 0;
    total = 0;
    inBlink = false;
    blinkStart = 1;
    
    % Add sentinel value
    actual = [actual(:); categorical({'n/a'})];
    
    for i = 1:length(actual)-1
        if actual(i) == 'blink'
            if ~inBlink
                total = total + 1;
                inBlink = true;
                blinkStart = i;
            end
        elseif inBlink && actual(i) ~= 'blink'
            % Process completed blink
            blinkEnd = i-1;
            blinkRegion = blinkStart:blinkEnd;
            
            if ~isempty(blinkRegion)
                [~, maxLoc] = max(signal(blinkRegion));
                peakLoc = blinkStart + maxLoc - 1;
                predBlinks = find(predicted == 'blink');
                
                if ~isempty(predBlinks) && ...
                   peakLoc >= min(predBlinks) && ...
                   peakLoc <= max(predBlinks)
                    correct = correct + 1;
                end
            end
            inBlink = false;
        end
    end
end