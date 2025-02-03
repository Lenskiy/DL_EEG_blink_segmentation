classdef scbt2ssscbLayer < nnet.layer.Layer ... 
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
            numSpatialDim = X.size(idx);
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

            % X = dlarray(X,'UUU');
            % Z = ones(numSamples, 1, numChannels, numBatches);
            % Z(:,1,:,:) = permute(X, [3, 1, 2]);
            % Z = dlarray(Z,'SSCB');

            X= dlarray(X,[repelem('U',numSpatialDim), "UUU"]);
            switch(numSpatialDim)
                case 1
                    Z(1:numSamples,1,1:numChannels,1:numBatches) = X.permute([1, 4, 2, 3]);
                    Z = dlarray(Z,'SSCB');
                case 2
                    Z(1:numSamples,1,1:numChannels,1:numBatches) = X.permute([1, 2, 5, 3, 4]);
                    Z = dlarray(Z,'SSSCB');
            end
            
        end
    end
end