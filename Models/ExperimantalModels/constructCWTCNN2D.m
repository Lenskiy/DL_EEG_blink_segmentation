function lgraph = constructCWTCNN2D(parameters)
    spatialDim = 64;
    layer  = [sequenceInputLayer([spatialDim parameters.numChannels],  Name="sequenceInputLayer", MinLength=128)
              functionLayer(@(x) repelem(x, 1, parameters.numFilters), Name="repeatInputLayer")];
    lgraph = layerGraph(layer);
    outputName = lgraph.Layers(end).Name;
    
    for k = 1:parameters.numBlocks
        lgraph = addLayers(lgraph, depthConcatenationLayer(parameters.numChannels*parameters.numFilters/2^(k-1), Name="dc_" + k));
        for l=1:(parameters.numChannels*parameters.numFilters/2^(k-1)) 
                layers = [functionLayer(@(z) z(:,l,:,:),'Name',"channels_" + k + "_" + l)
                          convolution2dLayer([min(parameters.filterSize, ceil(spatialDim/4^(k-1))) parameters.filterSize], 1,...
                               PaddingValue="replicate", Padding="same", Name="conv_" + k + "_" + l)];
                
                lgraph = addLayers(lgraph, layers);
                lgraph = connectLayers(lgraph, outputName, "channels_" + k + "_" + l);
                lgraph = connectLayers(lgraph, "conv_" + k + "_" + l,  "dc_" + k + "/in" + l);
        end

        layers = [
            convolution1dLayer(1, parameters.numChannels*parameters.numFilters,'Stride', 1 ,'Padding','same', Name="channel_pooling_" + k)
            batchNormalizationLayer(Name="bn_" + k)
            swishLayer(Name="swish_" + k) %maxPooling1dLayer(2, Stride=2, Name="pooling_"+k)
            averagePooling2dLayer([4, 1], Name="avg" + k, Stride=[4, 1])
            spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_"+k)
            ];

        lgraph = addLayers(lgraph, layers);
        lgraph = connectLayers(lgraph, "dc_" + k, "channel_pooling_" + k);

        outputName = "dropout_" + k;
    end
    
    lgraph= removeLayers(lgraph, "dropout_"+k);
    
    outputName = lgraph.Layers(end).Name;
    layers = [
            globalAveragePooling1dLayer(Name='avg_layer')
            flattenLayer(Name="flatten")
            fullyConnectedLayer(parameters.numLabels,'Name','fc')
            softmaxLayer
            classificationLayer];                
    
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, outputName, 'avg_layer');
    %analyzeNetwork(lgraph)
end