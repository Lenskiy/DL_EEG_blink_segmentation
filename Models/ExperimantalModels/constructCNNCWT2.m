function lgraph = constructCNNCWT2(parameters)
    samplingFreq = 128;
% numFeatures,numBlocks,filterSize,numFilters,poolSize,numResponse
    layer  = [sequenceInputLayer(parameters.numChannels,  Name="sequenceInputLayer") % MinLength=signalLength,
              cwtLayer(Name="CWT", FrequencyLimits=[0.25/samplingFreq 12/samplingFreq], VoicesPerOctave=8, Wavelet="Morse", TransformMode="mag", SignalLength=)]; % , SignalLength=signalLength
    
    lgraph = layerGraph(layer);
    lgraph = addLayers(lgraph, functionLayer(@(x) repelem(x, 1, parameters.numFilters, 1), Name="repeatInputLayer"));
    lgraph = connectLayers(lgraph, "CWT", "repeatInputLayer");

    outputName = lgraph.Layers(end).Name;
    
    for k = 1:parameters.numBlocks
        
        lgraph = addLayers(lgraph, depthConcatenationLayer(parameters.numChannels * parameters.numFilters, Name="dc_" + k));
        for l=1:parameters.numChannels * parameters.numFilters
                lgraph = addLayers(lgraph, functionLayer(@(z) z(:,l,:,:),'Name',"channels_" + k + "_" + l));
                lgraph = connectLayers(lgraph, outputName, "channels_" + k + "_" + l);

                lgraph = addLayers(lgraph, convolution2dLayer([ceil(parameters.filterSize/2^(k-1)) parameters.filterSize], 1,...
                    PaddingValue=0, Padding="same", Name="conv_" + k + "_" + l));
                lgraph = connectLayers(lgraph,  "channels_" + k + "_" + l,  "conv_" + k + "_" + l);

                lgraph = connectLayers(lgraph, "conv_" + k + "_" + l,  "dc_" + k + "/in" + l);
        end


        layers = [
            batchNormalizationLayer(Name="bn_" + k)
            reluLayer(Name="relu_" + k) %maxPooling1dLayer(2, Stride=2, Name="pooling_"+k)
            averagePooling2dLayer([2, 1], Name="avg" + k, Stride=[2, 1])
            spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_"+k)
            ];

        lgraph = addLayers(lgraph, layers);
        lgraph = connectLayers(lgraph, "dc_" + k, "bn_" + k);

        outputName = "dropout_" + k;
    end
    
    %lgraph = removeLayers(lgraph, "pooling_" + parameters.numBlocks);
    outputName = lgraph.Layers(end).Name;
    layers = [ % averagePooling2dLayer([23 1], Stride=1, Name="avg")% convolution2dLayer([23 1], 1,'Padding',[0 0 0 0],"Name",'fullCov') %convolution1dLayer(23, parameters.numLabels, Name='fullCov')
            functionLayer(@(x) sum(x, 1), Name="sum_layer")
            flattenLayer(Name="flatten")
            fullyConnectedLayer(parameters.numLabels,'Name','fc')
            softmaxLayer
            classificationLayer];                
    
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, outputName, 'sum_layer');
    %analyzeNetwork(lgraph)
end