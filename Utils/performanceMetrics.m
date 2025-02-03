function score = performanceMetrics(predicted, actual, scoreType, signal)
% Evaluate the accuracy of model prediction with scoreType 'f1Score' or 'f2Score' or
% 'blinkInRange' or 'blinkFinded'.
% accuracy = costumAccuracyCalculate(predictedY, groundTruthY, scoreType, signal)
% If f1 or f2 score apply, then signal is not nessesary.
    arguments
    predicted, actual,
    scoreType = 'f1Score',
    signal = []
    end

    % Compute true positives
    tp = 0;
    for k = 1:length(predicted)
        tp = tp + inRangeBlink(predicted{k}, actual{k}, signal{k});
    end

    if strcmp(scoreType, 'accuracy')
        [~, recall] = computePrecisionRecall(predicted, actual);
        score = recall;
        % accuracy        = 100*mean(cellfun(@(y1, y2) sum(y1 == y2)/length(y1), predicted, actual));
     elseif strcmp(scoreType, 'f1Score')
        [precision, recall] = computePrecisionRecall(predicted, actual);
        score = computeF1Score(precision, recall);
    elseif strcmp(scoreType, 'f2Score')
        [precision, recall] = computePrecisionRecall(predicted, actual);
        score = computeF2Score(precision, recall);
    elseif strcmp(scoreType, 'blinkInRange')
        % if the ground truth peak is located in 'blink' region. Still have
        % precision and recall... How to solve this?

        %tp = sum(cellfun(@(p, a, s) inRangeBlink(p, a, s), predicted, actual, signal));
        score = tp / sum(cellfun(@(a) blinkPiecesCount(a), actual));
    elseif strcmp(scoreType, 'blinkFound')
        % if a piece of detected signal correctly get something in it.
        
        %tp = sum(cellfun(@(a, p, s) inRangeBlink(a, p, s), actual, predicted, signal));

        
        score = tp / sum(cellfun(@(p) blinkPiecesCount(p), predicted));
        % double blink problem..
    elseif strcmp(scoreType, 'macroF1Score')
        %tp = sum(cellfun(@(p, a, s) inRangeBlink(p, a, s), predicted, actual, signal));
        actualCondition = sum(cellfun(@(a) blinkPiecesCount(a), actual));
        predictedCondition = sum(cellfun(@(p) blinkPiecesCount(p), predicted));
        precision = tp/predictedCondition;
        recall = tp/actualCondition;
        score = 2 * (precision * recall) / (precision + recall);
    else
        error('Invalid score type.');
    end
end
%%
function correctness = inRangeBlink(predicted, actual, signal)
    correctness = 0;
    inBlinkPiece = 0;
    for i = 1:length(actual)
        if actual(i) == 'blink'
            if ~inBlinkPiece
               inBlinkPiece = 1;
            end
        elseif inBlinkPiece && actual(i) ~= 'blink'        
            roi = find(actual == 'blink');
            if numel(roi) > 1
                [~, maxLocation] = max(signal(roi));
                location = roi(1) + maxLocation;
                pRoi = find(predicted == 'blink');
                if ~isempty(pRoi)
                    if location >= min(pRoi) && location <= max(pRoi)
                        correctness = correctness + 1;
                    end
                end
            end
            if inBlinkPiece
               inBlinkPiece = 0;
            end
        end
    end
end
%%
function [precision, recall] = computePrecisionRecall(predicted, actual)
    classes = categories(actual{1});
    
    precisionSum = 0;
    recallSum = 0;
    numClasses = length(classes);

    for i = 1:numClasses
        checking = classes{i, 1}; 
        tp = sum(cellfun(@(p, a) sum((p == checking) & (a == checking)), predicted, actual));
        fp = sum(cellfun(@(p, a) sum((p == checking) & (a ~= checking)), predicted, actual));
        fn = sum(cellfun(@(p, a) sum((p ~= checking) & (a == checking)), predicted, actual));
        
        if (tp + fp) > 0 && (tp + fn) > 0
            classPrecision = tp / (tp + fp);
            classRecall = tp / (tp + fn);
            precisionSum = precisionSum + classPrecision;
            recallSum = recallSum + classRecall;
        elseif tp == 0
            precisionSum = precisionSum + 1;
            recallSum = recallSum + 1;
        end
    end

    precision = precisionSum / numClasses;
    recall = recallSum / numClasses;
end
%% f-score
function f1Score = computeF1Score(precision, recall)
    f1Score = 2 * (precision * recall) / (precision + recall);
end

function f2Score = computeF2Score(precision, recall)
    f2Score = 5 * (precision * recall) / (4 * precision + recall);
end
%%
function blinkCount = blinkPiecesCount(signal)
    blinkCount = 0;
    inBlinkPiece = 0;

    for i = 1:length(signal)
        if signal(i) == "blink"
            if ~inBlinkPiece
                blinkCount = blinkCount + 1;
                inBlinkPiece = 1;
            end
        else
            % reset the flag
            inBlinkPiece = 0;
        end
    end
end