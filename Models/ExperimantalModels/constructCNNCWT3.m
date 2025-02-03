function lgraph = constructCNNCWT3(parameters)
    samplingFreq = 128;
    signalLength = 512;
% numFeatures,numBlocks,filterSize,numFilters,poolSize,numResponse
    layer  = [sequenceInputLayer(parameters.numChannels, MinLength=signalLength, Name="sequenceInputLayer")
              cwtLayer(Name="CWT", FrequencyLimits=[0.25/samplingFreq 6/samplingFreq], VoicesPerOctave=8, SignalLength=signalLength)];
    
    lgraph = layerGraph(layer);
    outputName = lgraph.Layers(end).Name;
    
    for k = 1:parameters.numBlocks
        layers = [
            convolution2dLayer([23 parameters.filterSize], parameters.numFilters,...
                PaddingValue=0, Padding="same", Name="conv_" + k)
            layerNormalizationLayer(Name="ln_" + k)
            swishLayer(Name="swish_" + k) %maxPooling1dLayer(2, Stride=2, Name="pooling_"+k)
            spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_"+k)
            ];

        lgraph = addLayers(lgraph, layers);
        lgraph = connectLayers(lgraph, outputName, "conv_" + k);

        outputName = "dropout_" + k;
    end
    
    %lgraph = removeLayers(lgraph, "pooling_" + parameters.numBlocks);
    outputName = lgraph.Layers(end).Name;
    layers = [ % averagePooling2dLayer([23 1], Stride=1, Name="avg")% convolution2dLayer([23 1], 1,'Padding',[0 0 0 0],"Name",'fullCov') %convolution1dLayer(23, parameters.numLabels, Name='fullCov')
            flattenLayer(Name="flatten")
            fullyConnectedLayer(parameters.numLabels,'Name','fc')
            softmaxLayer
            classificationLayer];                
    
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, outputName, 'flatten');
    analyzeNetwork(lgraph)
end