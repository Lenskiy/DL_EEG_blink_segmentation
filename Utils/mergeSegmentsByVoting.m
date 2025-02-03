function weightedLabeledEEGsegments = mergeSegmentsByVoting(labeledEEGsegments, recordsIDs, stepLenght)
% labeledEEGsegments: cell array with labelled segments. Each segment represent a porition of signals shifted by stepLenght
% For example, if the lenght of one segment is 384 and stepLenght is 128,
% then 2/3 of each segment overlaps with a previous or next segment.
% recordsIDs: records ID, this is used to separate the signals that were concatenated.
% stepLenght: is the step lenght, the overlap with adjacent segments oi
% segment lenght - step lenght
    uniqueSegmentIDs = unique(recordsIDs);
    rearrangedSegments{1}{1} = {};
    weightedLabeledEEGsegments= {};

    cats        = categories(labeledEEGsegments{1});
    numOfLabels = length(cats);
    priors      = hist(labeledEEGsegments{1});
    [~, defaultValue] = max(priors); % Use defaultValue for the case when there is a draw between the labels.

    for k = 1:length(uniqueSegmentIDs) % split the segments based on records IDs. This is neccessary to avoid a single segment containg signals from different records
         idx{k} = find(uniqueSegmentIDs(k) == recordsIDs);
         for l = 1:length(idx{k})
            rearrangedSegments{k}{l} = uint8(labeledEEGsegments{idx{k}(l)}); % unit8 is applied to convert categorical to integers
         end
    end

    % The loop below assumes that each segment is of 3*stepLenght
    % For example if stepLenght is 128 then segment lenght is 384
    for m = 1:length(rearrangedSegments)
        segments = [];
        segment  = [];
        for k = 3:length(rearrangedSegments{m}) + 2
            for l = 1:stepLenght
                if(k <= length(rearrangedSegments{m}))
                    labels = [rearrangedSegments{m}{k}(l); 
                              rearrangedSegments{m}{k-1}(stepLenght+l); 
                              rearrangedSegments{m}{k-2}(2*stepLenght+l)];
                elseif(k == length(rearrangedSegments{m}) + 1)
                    labels = [rearrangedSegments{m}{k-1}(stepLenght+l); 
                              rearrangedSegments{m}{k-2}(2*stepLenght+l)];
                else
                    labels = rearrangedSegments{m}{k-2}(2*stepLenght+l);
                end
    
                [counts, vals]= hist(labels, 1:numOfLabels);
                [val, ind] = max(counts);
        
                if(sum(val == counts) == length(counts))
                    %segment(l) = categorical(cats(defaultValue), cats); % in case there is a tie
                    segment(l) = defaultValue;
                else
                    %segment(l) = categorical(vals(ind), cats);
                    segment(l) = vals(ind);
                end
            end

            segments = [segments, segment];
            if(k - 2 == 1)
                %weightedLabeledEEGsegments{length(weightedLabeledEEGsegments) + 1} = categorical(rearrangedSegments{m}{k}, cats);
                weightedLabeledEEGsegments{idx{m}(k-2)} = renamecats(categorical([rearrangedSegments{m}{1}(1:2*stepLenght), segments], 1:numOfLabels), cats);
            elseif(k - 2 == 2)
                %weightedLabeledEEGsegments{length(weightedLabeledEEGsegments) + 1} = categorical([segments; rearrangedSegments{m}{k}(stepLenght+1:end)], cats);
                weightedLabeledEEGsegments{idx{m}(k-2)} = renamecats(categorical([rearrangedSegments{m}{2}(1:stepLenght), segments], 1:numOfLabels), cats);
            else
                %weightedLabeledEEGsegments{length(weightedLabeledEEGsegments) + 1} = categorical(segments, cats);
                weightedLabeledEEGsegments{idx{m}(k-2)} = renamecats(categorical(segments, 1:numOfLabels), cats);
                segments(1:stepLenght) = []; 
            end
        end
    end
end