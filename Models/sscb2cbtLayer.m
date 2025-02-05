classdef sscb2cbtLayer < nnet.layer.Layer & nnet.layer.Formattable 
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
            % Simple conversion from SSCB to CBT
            [~, ~, C, B] = size(X);  % Get channels and batch size
            Z = reshape(X, [], C, B); % Flatten spatial dimensions
            Z = permute(Z, [2 3 1]);  % Rearrange to CBT format
            Z = dlarray(Z, 'CBT');
        end
    end
end