classdef groupedConvolution1dLayer < nnet.layer.Layer ...
        & nnet.layer.Formattable ...
        & nnet.layer.Acceleratable ...
        & nnet.internal.cnn.layer.CPUFusableLayer
    
    % This layer performs grouped 1-D convolution on a CBT dlarray

    % Copyright 2023-2024 The MathWorks, Inc.

    %#codegen
    
    properties (Learnable)
        Weights
        Bias
    end
    properties (SetAccess=immutable)
        DilationFactor
        FilterSize
    end

    methods
        function layer = groupedConvolution1dLayer(options) 
            arguments
                options.Weights
                options.Bias
                options.FilterSize = 3
                options.DilationFactor = 1;
                options.Name = "";
            end
            layer.Name = options.Name;
            layer.Description = "Grouped 1D Convolution";

            if isfield(options,'Weights')
                layer.Weights = options.Weights;
            end
            if isfield(options,'Bias')
                layer.Bias = options.Bias;
            end
            layer.DilationFactor = options.DilationFactor;
            layer.FilterSize = options.FilterSize;
        end

        function layer = initialize(layer,layout)
            cdim = layout.Format=='C';
            csize = layout.Size(cdim);
            if isempty(layer.Weights)

                filterSize = layer.FilterSize;
                numChannelsPerGroup = 1;
                numFiltersPerGroup = 1;
                numGroups = csize;

                sz = [filterSize numChannelsPerGroup numFiltersPerGroup numGroups];
                numOut = filterSize * numFiltersPerGroup*numGroups;
                numIn = filterSize * numChannelsPerGroup*numGroups;

                layer.Weights = initializeGlorot(sz,numOut,numIn);
            end
            if isempty(layer.Bias)
                layer.Bias = dlarray(zeros([1,csize],'single'));
            end
        end
        function Z = predict(layer, X)
           Z = dlconv(X,layer.Weights,layer.Bias, ...
               WeightsFormat="TCUU", ...
               Padding="same", ...
               DilationFactor=layer.DilationFactor);
        end
    end

    % The following methods are required to enable CPU Fusable.
    methods(Hidden)
        function Z = cpuInferenceEnginePredict(this,X)
            %cpuInferenceEnginePredict Prediction method for the CPU
            % inference engine
            X = dlarray(X,"CBT");
            Z = this.predict(X);
            Z = extractdata(Z);
        end
        function layerArgs = getFusedArguments(this)
            %getFusedArguments  Return arguments needed to call the
            % layer in a fused network
            layerArgs = {'external', ...
                @this.cpuInferenceEnginePredict, ...
                1,1, ... % 1 in, 1 out
                [], ... % size(in)==size(out)
                2}; 
        end
        function tf = isFusable(~)
            %isFusable Flag if layer is fusable
            tf = true;
        end
    end
end

function weights = initializeGlorot(sz,numOut,numIn)

Z = 2*rand(sz,'single') - 1;
bound = sqrt(6 / (numIn + numOut));

weights = bound * Z;
weights = dlarray(weights);

end