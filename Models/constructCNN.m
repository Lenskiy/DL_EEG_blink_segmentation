function dlnet = constructCNN(parameters)

    layers  = [sequenceInputLayer(parameters.numChannels, Name="sequenceInputLayer", MinLength=128), 
               cbt2sscbLayer(Name="t2s")]; % Normalization removed 2023.07
    lgraph = layerGraph(layers);
    
    outputName = layers(end).Name;
    
    for k = 1:parameters.numBlocks
        if(k == 1)
            layers = [
                convolution2dLayer([parameters.filterSize 1], parameters.numFilters, Padding="same", Name="conv_" + k)
                layerNormalizationLayer(Name="ln_" + k)
                swishLayer(Name="swish_" + k)
                spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_" + k)
                ];
        else
            layers = [
                groupedConvolution2dLayer([parameters.filterSize, 1], 1, 'channel-wise', Padding="same", Name="conv_" + k)
                convolution2dLayer(1, parameters.numFilters, Name="conv_fc_" + k)
                layerNormalizationLayer(Name="ln_" + k)
                swishLayer(Name="swish_" + k)
                spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_" + k)
                ];
        end
        lgraph = addLayers(lgraph, layers);
        lgraph = connectLayers(lgraph, outputName, "conv_" + k);
        outputName = "dropout_" + k;
    end
    
    layers = [
        sscb2cbtLayer(Name="s2t")
        flattenLayer(Name="fl")
        fullyConnectedLayer(parameters.numLabels, Name="fc")
        softmaxLayer
        ];
    
    lgraph = addLayers(lgraph,layers);
    lgraph = connectLayers(lgraph, outputName, "s2t");
    dlnet  = dlnetwork(lgraph);
end