classdef sscb2cbtLayer < nnet.layer.Layer ... 
        & nnet.layer.Formattable 

    properties
        OutputSize
    end
    
    methods
        function layer = sscb2cbtLayer(NameValueArgs)
            arguments
                NameValueArgs.Name = 'ConvertSpatialToTime';
            end
            
            layer.Name = NameValueArgs.Name;
            layer.Description = "Convert Spatial To Time";
            layer.Type = "Convert Spatial To Time";
        end

        function layer = initialize(layer, ~)
            layer.OutputSize = [];
        end

        function Z = predict(~, X)
            % Get dimensions
            idx = finddim(X, "S");
            numSpatialDims = length(idx);
            spatialSizes = zeros(1, numSpatialDims);
            for k = 1:numSpatialDims
                spatialSizes(k) = X.size(idx(k));
            end
            
            idx = finddim(X, "C");
            numChannels = X.size(idx);
            
            idx = finddim(X, "B");
            numBatches = 1;
            if ~isnan(X.size(idx))
                numBatches = X.size(idx);
            end
            
            % Process based on number of spatial dimensions
            if numSpatialDims == 1
                % Single spatial dimension case
                Z = dlarray(X, 'CBT');
            elseif numSpatialDims == 2
                if any(spatialSizes == 1)
                    % Case where one spatial dimension is 1
                    % Reshape to remove the unit dimension
                    effectiveSize = max(spatialSizes);
                    Z = dlarray(reshape(X, [numChannels, numBatches, effectiveSize]), 'CBT');
                else
                    % Both spatial dimensions are significant
                    Z = dlarray(X, 'SCBT');
                end
            else
                error('Unexpected number of spatial dimensions');
            end
        end
    end
end