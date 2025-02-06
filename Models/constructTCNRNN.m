function dlnet = constructTCNRNN(parameters)
    % Initial layers
    layers = [sequenceInputLayer(parameters.numChannels, Name="sequenceInputLayer", MinLength=128)
              cbt2sscbLayer(Name="t2s")];
    lgraph = layerGraph(layers);
    outputName = layers(end).Name;
    
    % TCN blocks with dilated convolutions
    for k = 1:parameters.numBlocks
        dilationFactor = 2^(k-1);
        if(k == 1)
            layers = [
                convolution2dLayer([parameters.filterSize, 1], parameters.numFilters, DilationFactor=dilationFactor, Padding="same", Name="conv_1_" + k)
                layerNormalizationLayer
                spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_" + k)
                groupedConvolution2dLayer([parameters.filterSize, 1], 1, 'channel-wise', DilationFactor=dilationFactor, Padding="same")
                convolution2dLayer(1, parameters.numFilters)
                layerNormalizationLayer
                swishLayer
                spatialDropoutLayer(parameters.dropoutFactor)
                additionLayer(2,Name="add_" + k)];
        else
            layers = [
                groupedConvolution2dLayer([parameters.filterSize, 1], 1, 'channel-wise', DilationFactor=dilationFactor, Padding="same", Name="conv_1_" + k)
                convolution2dLayer(1, parameters.numFilters, Name="conv_2_" + k)
                layerNormalizationLayer
                spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_" + k)
                groupedConvolution2dLayer([parameters.filterSize, 1], 1, 'channel-wise', DilationFactor=dilationFactor, Padding="same")
                convolution2dLayer(1, parameters.numFilters)
                layerNormalizationLayer
                swishLayer
                spatialDropoutLayer(parameters.dropoutFactor)
                additionLayer(2,Name="add_"+k)];        
        end

        % Add and connect layers
        lgraph = addLayers(lgraph,layers);
        lgraph = connectLayers(lgraph,outputName,"conv_1_"+k);

        % Skip connection
        layer = convolution2dLayer(1, parameters.numFilters,Name="convSkip_" + k);
        lgraph = addLayers(lgraph,layer);
        lgraph = connectLayers(lgraph, outputName,"convSkip_" + k);
        lgraph = connectLayers(lgraph, "convSkip_" + k,"add_" + k + "/in2");
        
        outputName = "add_" + k;
    end
    
    % Convert format for RNN
    layers = [
        sscb2cbtLayer(Name="s2t")
        flattenLayer(Name="fl")];
    
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, outputName, "s2t");
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