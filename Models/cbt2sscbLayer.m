classdef cbt2sscbLayer < nnet.layer.Layer ... 
        & nnet.layer.Formattable 

    properties
        % Output size
        OutputSize
    end
    
    methods
        function layer = cbt2sscbLayer(NameValueArgs)
            arguments
                NameValueArgs.Name = 'ConvertTimeToSpatial';
            end

            layer.Name = NameValueArgs.Name;

            layer.Description = "Convert Time To Spatial";
           
            layer.Type = "Convert Time To Spatial";

        end

        function layer = initialize(layer, ~)
            layer.OutputSize = [];
        end

        function Z = predict(~, X)
            idx = finddim(X, "S");
            if(~isempty(idx))
                numSpatialDimensions = X.size(idx(1));
            else
                numSpatialDimensions = 0;
            end
            idx = finddim(X, "T");
            numSamples = X.size(idx(1));
            idx = finddim(X, "C");
            numChannels = X.size(idx);
            idx = finddim(X, "B");
            if(isnan(X.size(idx)))
                numBatches = 1;
            else
                numBatches = X.size(idx); 
            end

            if(numSpatialDimensions == 0)
               X = dlarray(X,'UUU');
               Z(1:numSamples,1,1:numChannels,1:numBatches) = X.permute([3, 1, 2]);
            else
               X = dlarray(X,'UUUU');
               Z(1:numSpatialDimensions,1:numSamples,1:numChannels,1:numBatches) = X.permute([1, 4, 2, 3]);
            end
            Z = dlarray(Z,'SSCB');
        end
    end
end