function [searchTrain, testSave, validationTest] = splitSignalIntoSeveralSegments(signals, datasetSplitRatio, step, segLength)

    if(size(signals, 1) < size(signals, 2))
        signals = signals';
    end

    numSegments = floor((height(signals) - segLength)/step + 1);

    searchTrainRatio    = datasetSplitRatio(1); 
    testSaveRatio       = datasetSplitRatio(2);
    validationTestRatio = datasetSplitRatio(3);
    

    searchTrainIndex    = sort(randperm(numSegments, round(searchTrainRatio*numSegments)));
    leftIndex           = setdiff(1:numSegments, searchTrainIndex);
    testSaveIndexIndex  = sort(randperm(numel(leftIndex), round(testSaveRatio/(1-searchTrainRatio)*numel(leftIndex))));
    testSaveIndex       = leftIndex(testSaveIndexIndex);
    validationTestIndex = setdiff(1:numSegments, [searchTrainIndex testSaveIndex]);

    [~, searchTrain]       = splitSignalIntoSegmentsWithIndexes(signals, searchTrainIndex,    step, segLength);
    [~, testSave]          = splitSignalIntoSegmentsWithIndexes(signals, testSaveIndex,       step, segLength);
    [~, validationTest]    = splitSignalIntoSegmentsWithIndexes(signals, validationTestIndex, step, segLength);

end

function [setOfSegments, indexOfSet] = splitSignalIntoSegmentsWithIndexes(signal, indexes, step, segLength)
    setOfSegments = [];
    indexOfSet = [];
    k = 1;
    while(k <= length(indexes))
%         k
        l = 0;
        while(k + l + 1 < length(indexes) && (indexes(k + l + 1) - indexes(k+l)) == 1)
          l = l + 1;
        end
    
        segmentsTemp    = zeros((indexes(k+l)-indexes(k) + 1) * round(1.5*segLength/step) -  segLength/step, size(signal,2), segLength);
        indexesTemp     = [];
        offset = (indexes(k)-1) * round(1.5*segLength);
        for m = 0:(indexes(k+l)-indexes(k) + 1) * round(1.5*segLength/step) -  segLength/step %(length(tempSegment)/step - segLength/step)
            indexesTemp(m + 1, :) = offset + ((m*step+1):m*step+segLength);
            if(istable(signal))
                segmentsTemp(m + 1, :, :) = signal(indexesTemp(m + 1, :), :).Variables';
            else
                segmentsTemp(m + 1, :, :) = signal(indexesTemp(m + 1, :), :);
            end
        end

        setOfSegments   = [setOfSegments; segmentsTemp];
        indexOfSet         = [indexOfSet; indexesTemp];
        k = k + l + 1;
    end
end