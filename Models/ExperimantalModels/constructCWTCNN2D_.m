function lgraph = constructCWTCNN2D_(parameters)
    spatialDim = 64;
    layer  = [sequenceInputLayer([spatialDim parameters.numChannels],  Name="sequenceInputLayer", MinLength=128)
              ]; % cbt2sscbLayer(Name="t2s")
    lgraph = layerGraph(layer);
    
    outputName = layer(end).Name;
    
    for k = 1:parameters.numBlocks
        if(k == 1)
            layers = [
                cbt2sscbLayer(Name="c2s_" + k)
                convolution2dLayer([parameters.filterSize, parameters.filterSize], parameters.numFilters, Padding="same", Name="conv_" + k, WeightsInitializer="narrow-normal")
                sscb2cbtLayer(Name="s2c_" + k)
                layerNormalizationLayer
                spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_" + k)
                swishLayer(Name="swish_" + k) % averagePooling2dLayer([1, 4], Name="avg" + k, Stride=[1, 4])
                ];
        else
            layers = [
                cbt2sscbLayer(Name="c2s_" + k)
                groupedConvolution2dLayer([parameters.filterSize, parameters.filterSize], 1, 'channel-wise', Padding="same", Name="conv_" + k)
                convolution2dLayer(1, parameters.numFilters, Name="channel_mixing_" + k)
                sscb2cbtLayer(Name="s2c_" + k)
                layerNormalizationLayer
                spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_" + k)
                swishLayer(Name="swish_" + k) %averagePooling2dLayer([1, 4], Name="avg" + k, Stride=[1, 4])
                ];           
        end

        lgraph = addLayers(lgraph, layers);
        lgraph = connectLayers(lgraph, outputName, "c2s_" + k);
        outputName = "swish_" + k;
    end
    
    %lgraph= removeLayers(lgraph, "dropout_"+k);
    
    outputName = lgraph.Layers(end).Name;
    layers = [ % sscb2cbtLayer(Name="s2t")
            globalAveragePooling1dLayer(Name='avg_layer')
            flattenLayer(Name="flatten")
            fullyConnectedLayer(parameters.numLabels,'Name','fc')
            softmaxLayer
            classificationLayer];                
    
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, outputName, 'avg_layer');
    analyzeNetwork(lgraph)
end