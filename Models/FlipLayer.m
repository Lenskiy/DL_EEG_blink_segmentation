classdef FlipLayer < nnet.layer.Layer
    methods
        function layer = FlipLayer(NameValueArgs)
            arguments
                NameValueArgs.Name = ''
            end
            layer.Name = NameValueArgs.Name;
        end
        function Y = predict(~, X)
            Y = flip(X,3);
        end
    end
end