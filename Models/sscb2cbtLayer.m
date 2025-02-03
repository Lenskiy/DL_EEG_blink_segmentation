classdef sscb2cbtLayer < nnet.layer.Layer ... 
        & nnet.layer.Formattable 

    properties
        % Output size
        OutputSize
    end
    
    methods
        function layer = sscb2cbtLayer(NameValueArgs)
            arguments
                NameValueArgs.Name = 'ConvertSpatialToTime';
            end
            
            name = NameValueArgs.Name;

            layer.Name = name;

            layer.Description = "Convert Spatial To Time";
           
            layer.Type = "Convert Spatial To Time";

        end

        function layer = initialize(layer, ~)
            layer.OutputSize = [];
        end

        function Z = predict(~, X)
            idx = finddim(X, "S");
            for k = 1:length(idx)
                numSamples(k) = X.size(idx(k));
            end
            idx = finddim(X, "C");
            numChannels = X.size(idx);
            idx = finddim(X, "B");
            if(isnan(X.size(idx)))
                numBatches = 1;
            else
                numBatches = X.size(idx); 
            end
            
            switch size(numSamples, 2)
            case 1
                X = X.reshape([numSamples, numChannels, numBatches]);
                X = dlarray(X,'UUU');
                Z = X.permute([2 3 1]);
                Z = dlarray(Z,'CBT');
            case 2
                %X = X.reshape([numSamples, numChannels, numBatches]);
                X = dlarray(X,'UUUU');
                Z = X.permute([2 3 4 1]);
                Z = dlarray(Z,'SCBT');
            end
        end
    end
end