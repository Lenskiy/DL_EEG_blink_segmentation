function dlnet = constructTCNsRNN(parameters)
    % Initial layers
    layers = [sequenceInputLayer(parameters.numChannels, Name="sequenceInputLayer", MinLength=128)];
    lgraph = layerGraph(layers);
    outputName = layers(end).Name;
    
    % TCN blocks with standard dilated convolutions
    for k = 1:parameters.numBlocks
        dilationFactor = 2^(k-1);
        
        layers = [
            convolution1dLayer(parameters.filterSize, parameters.numFilters, ...
                DilationFactor=dilationFactor, Padding="causal", Name="conv1_" + k)
            layerNormalizationLayer
            spatialDropoutLayer(parameters.dropoutFactor)
            convolution1dLayer(parameters.filterSize, parameters.numFilters, ...
                DilationFactor=dilationFactor, Padding="causal")
            layerNormalizationLayer
            swishLayer
            spatialDropoutLayer(parameters.dropoutFactor)
            additionLayer(2, Name="add_" + k)
        ];
    
        % Add and connect layers
        lgraph = addLayers(lgraph, layers);
        lgraph = connectLayers(lgraph, outputName, "conv1_" + k);
    
        % Skip connection
        if k == 1
            % Include convolution in first skip connection
            layer = convolution1dLayer(1, parameters.numFilters, Name="convSkip");
            lgraph = addLayers(lgraph, layer);
            lgraph = connectLayers(lgraph, outputName, "convSkip");
            lgraph = connectLayers(lgraph, "convSkip", "add_" + k + "/in2");
        else
            lgraph = connectLayers(lgraph, outputName, "add_" + k + "/in2");
        end
        
        outputName = "add_" + k;
    end
    
    % Convert format for RNN (no format conversion needed as we're using 1D convolutions)
    layers = [flattenLayer(Name="fl")];
    
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, outputName, "fl");
    outputName = layers(end).Name;

    % Add recurrent layers
    for k = 1:parameters.numRBlocks
        layerOrLayerGraph = eval(parameters.typeOfUnit + "(" + num2str((parameters.numUnits/2^(k-1)) + "," + "Name=['rnn_', num2str(" + k + ")])"));

        if(class(layerOrLayerGraph) == "nnet.cnn.LayerGraph")
            lgraph = mergeNetworkGraphs(lgraph, layerOrLayerGraph);
            lgraph = connectLayers(lgraph, outputName, "rnn_"+k+"_input");
        else
            lgraph = addLayers(lgraph,layerOrLayerGraph);
            lgraph = connectLayers(lgraph, outputName, "rnn_"+k);
        end

        layers = [dropoutLayer(parameters.dropoutFactor, Name="dropout_recurrent_" + k)];
        lgraph = addLayers(lgraph,layers);
        lgraph = connectLayers(lgraph, "rnn_"+k, layers(end).Name);
        
        outputName = "dropout_recurrent_" + k;
    end

    lgraph = removeLayers(lgraph, "dropout_recurrent_" + k);

    % Output layers
    layers = [fullyConnectedLayer(parameters.numLabels,'Name','fc')
              softmaxLayer];    

    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, "rnn_" + k, "fc");
    dlnet = dlnetwork(lgraph);
end