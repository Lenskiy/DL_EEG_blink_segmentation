function weightedLabeledSegments = mergeSegmentsByVoting2(labeledEEGsegments, recordsIDs, stepLenght)
% labeledEEGsegments: cell array with labelled segments. Each segment represent a porition of signals shifted by stepLenght
% For example, if the lenght of one segment is 384 and stepLenght is 128,
% then 2/3 of each segment overlaps with a previous or next segment.
% recordsIDs: records ID, this is used to separate the signals that were concatenated.
% stepLenght: is the step lenght, the overlap with adjacent segments oi
% segment lenght - step lenght
    uniqueSegmentIDs = unique(recordsIDs);
    rearrangedSegments{1}{1} = {};
    weightedLabeledSegments= {};

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


    offsetInSegments = ceil(length(labeledEEGsegments{1})/stepLenght);
    % The loop below assumes that each window consists of
    % "offsetInSegments" segments
    % For example if stepLenght is 128 and segment lenght is 384 then
    % offsetInSegments is 3

    for m = 1:length(rearrangedSegments)
        segments = [];
        segment  = [];
        for k = offsetInSegments:length(rearrangedSegments{m}) + (offsetInSegments - 1)
            for l = 1:stepLenght
                labels = [];
                for n = 0:(offsetInSegments - 1)
                    tempSegment = rearrangedSegments{m}{min(k-n, length(rearrangedSegments{m}))};
                    labels = [labels; tempSegment(min(l + n*stepLenght, length(tempSegment)))];
                end

                [counts, vals]= hist(labels, 1:numOfLabels);
                [val, ind] = max(counts);

                if(sum(val == counts) == length(counts))
                    segment(l) = defaultValue;
                else
                    segment(l) = vals(ind);
                end
            end

            segments = [segments, segment];
            if(k - offsetInSegments + 1 < offsetInSegments) % the case that deals with segments that don't have enough overlaps i.e. the first (offsetInSegments - 1) segments.
                weightedLabeledSegments{idx{m}(k - offsetInSegments + 1)} = renamecats(categorical([rearrangedSegments{m}{1}(1:(offsetInSegments - (k - offsetInSegments + 1))*stepLenght), segments], 1:numOfLabels), cats);
            else
                weightedLabeledSegments{idx{m}(k - offsetInSegments + 1)} = renamecats(categorical(segments, 1:numOfLabels), cats);
                segments(1:stepLenght) = [];
            end
        end
    end
end

% OLD Implementation
% function mergedLabels = mergeSegmentsByVoting2(labeledSegments, recordIDs, stepLength)
%     % MERGESBEGMENTSBYVOTING2 Combines overlapping segment predictions using majority voting
%     %
%     % Inputs:
%     %   labeledSegments - Cell array of categorical arrays (each 1x384)
%     %   recordIDs      - Array of record IDs to separate different patients' data
%     %   stepLength     - Step size between segments (64)
% 
%     % Get basic parameters
%     segmentLength = length(labeledSegments{1});  % 384
%     offsetInSegments = ceil(segmentLength/stepLength);  % 6
%     cats = categories(labeledSegments{1});
%     numCategories = length(cats);
% 
%     % Split segments by record ID and convert to numeric
%     uniqueRecordIDs = unique(recordIDs);
%     mergedLabels = cell(1, length(labeledSegments));
% 
%     % Process each record separately
%     for recordIdx = 1:length(uniqueRecordIDs)
%         % Get segments for current record
%         currentIndices = find(recordIDs == uniqueRecordIDs(recordIdx));
%         currentSegments = cellfun(@(x) uint8(x), labeledSegments(currentIndices), 'UniformOutput', false);
% 
%         segments = [];
%         segment = [];
% 
%         % Process segments with voting
%         for k = offsetInSegments:length(currentSegments) + (offsetInSegments - 1)
%             % Process each position in step length
%             for l = 1:stepLength
%                 % Collect votes from overlapping segments
%                 votes = [];
%                 for n = 0:(offsetInSegments - 1)
%                     segIdx = min(k-n, length(currentSegments));
%                     if segIdx > 0
%                         pos = min(l + n*stepLength, length(currentSegments{segIdx}));
%                         votes = [votes; currentSegments{segIdx}(pos)];
%                     end
%                 end
% 
%                 % Perform majority voting
%                 [counts, vals] = hist(votes, 1:numCategories);
%                 [maxCount, maxIdx] = max(counts);
% 
%                 % Handle ties using first category with max count
%                 segment(l) = vals(maxIdx);
%             end
% 
%             segments = [segments, segment];
% 
%             % Handle start of record
%             if (k - offsetInSegments + 1) < offsetInSegments
%                 startLength = (offsetInSegments - (k - offsetInSegments + 1))*stepLength;
%                 fullSegment = [currentSegments{1}(1:startLength), segments];
%                 mergedLabels{currentIndices(k - offsetInSegments + 1)} = categorical(fullSegment, 1:numCategories, cats);
%             else
%                 mergedLabels{currentIndices(k - offsetInSegments + 1)} = categorical(segments, 1:numCategories, cats);
%                 segments(1:stepLength) = [];
%             end
%         end
%     end
% end
