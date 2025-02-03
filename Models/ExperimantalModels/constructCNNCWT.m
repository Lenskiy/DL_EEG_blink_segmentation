function lgraph = constructCNNCWT(parameters)
    samplingFreq = 128;
    signalLength = 512;
% numFeatures,numBlocks,filterSize,numFilters,poolSize,numResponse
    layer  = [sequenceInputLayer(parameters.numChannels, MinLength=signalLength, Name="sequenceInputLayer")
              cwtLayer(Name="CWT", FrequencyLimits=[0.25/samplingFreq 6/samplingFreq], VoicesPerOctave=8, SignalLength=signalLength)
              sequenceFoldingLayer('Name','fold')];
    
    lgraph = layerGraph(layer);
    lgraph = addLayers(lgraph, functionLayer(@(x) repelem(x, 1, parameters.numFilters, 1), Name="repeatInputLayer"));
    lgraph = connectLayers(lgraph, "fold/out", "repeatInputLayer");

    outputName = lgraph.Layers(end).Name;
    
    for k = 1:parameters.numBlocks
        
        lgraph = addLayers(lgraph, depthConcatenationLayer(parameters.numChannels * parameters.numFilters, Name="dc_" + k));
        for l=1:parameters.numChannels * parameters.numFilters
                lgraph = addLayers(lgraph, functionLayer(@(z) z(:,l,:),'Name',"channels_" + k + "_" + l));
                lgraph = connectLayers(lgraph, outputName, "channels_" + k + "_" + l);

                lgraph = addLayers(lgraph, convolution1dLayer(parameters.filterSize, 1,...
                    PaddingValue="replicate", Padding="same", Name="conv_" + k + "_" + l));
                lgraph = connectLayers(lgraph,  "channels_" + k + "_" + l,  "conv_" + k + "_" + l);

                lgraph = connectLayers(lgraph, "conv_" + k + "_" + l,  "dc_" + k + "/in" + l);
        end


        layers = [
            layerNormalizationLayer(Name="ln_" + k)
            swishLayer(Name="swish_" + k) %maxPooling1dLayer(2, Stride=2, Name="pooling_"+k)
            spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_"+k)
            ];v

        lgraph = addLayers(lgraph, layers);
        lgraph = connectLayers(lgraph, "dc_" + k, "ln_" + k);

        outputName = "dropout_" + k;
    end
    
    %lgraph = removeLayers(lgraph, "pooling_" + parameters.numBlocks);
    outputName = lgraph.Layers(end).Name;
    layers = [ % averagePooling2dLayer([23 1], Stride=1, Name="avg")% convolution2dLayer([23 1], 1,'Padding',[0 0 0 0],"Name",'fullCov') %convolution1dLayer(23, parameters.numLabels, Name='fullCov')
            sequenceUnfoldingLayer('Name','unfold')
            flattenLayer(Name="flatten")
            fullyConnectedLayer(parameters.numLabels,'Name','fc')
            softmaxLayer
            classificationLayer];                
    
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, outputName, 'unfold/in');
    lgraph = connectLayers(lgraph, 'fold/miniBatchSize','unfold/miniBatchSize');
    %analyzeNetwork(lgraph)
end