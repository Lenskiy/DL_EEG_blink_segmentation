function lgraph = constructCWTCNN3D(parameters)
    spatialDim = 64;
    layer  = [sequenceInputLayer([spatialDim parameters.numChannels 1],  Name="sequenceInputLayer", MinLength=128)
              functionLayer(@(x) repelem(x, 1, 1, parameters.numFilters), Name="repeatInputLayer")];
    lgraph = layerGraph(layer);
    outputName = lgraph.Layers(end).Name;
    
    for k = 1:parameters.numBlocks
        lgraph = addLayers(lgraph, depthConcatenationLayer(parameters.numFilters/2^(k-1), Name="dc_" + k));
        for l=1:(parameters.numFilters/2^(k-1)) 
                layers = [functionLayer(@(z) z(:,:,l,:,:),'Name',"channels_" + k + "_" + l)
                          convolution3dLayer([min(parameters.filterSize, ceil(spatialDim/4^(k-1))) parameters.filterSize parameters.numChannels], 1,...
                               PaddingValue="replicate", Padding="same", Name="conv_" + k + "_" + l)];
                
                lgraph = addLayers(lgraph, layers);
                lgraph = connectLayers(lgraph, outputName, "channels_" + k + "_" + l);
                lgraph = connectLayers(lgraph, "conv_" + k + "_" + l,  "dc_" + k + "/in" + l);
        end

        layers = [
            convolution2dLayer(1, parameters.numFilters,'Stride', 1 ,'Padding','same', Name="channel_pooling_" + k)
            batchNormalizationLayer(Name="bn_" + k)
            swishLayer(Name="swish_" + k) %maxPooling1dLayer(2, Stride=2, Name="pooling_"+k)
            averagePooling3dLayer([4, 1, 1], Name="avg" + k, Stride=[4, 1, 1])
            spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_"+k)
            ];

        lgraph = addLayers(lgraph, layers);
        lgraph = connectLayers(lgraph, "dc_" + k, "channel_pooling_" + k);

        outputName = "dropout_" + k;
    end
    
    lgraph= removeLayers(lgraph, "dropout_"+k);
    
    outputName = lgraph.Layers(end).Name;
    layers = [
            globalAveragePooling2dLayer(Name='avg_layer')
            flattenLayer(Name="flatten")
            fullyConnectedLayer(parameters.numLabels,'Name','fc')
            softmaxLayer
            classificationLayer];                
    
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, outputName, 'avg_layer');
    %analyzeNetwork(lgraph)
end
% function lgraph = constructCWTCNN3D(parameters)
%     spatialDim = 64;
%     layer  = [sequenceInputLayer([spatialDim parameters.numChannels 1],  Name="sequenceInputLayer", MinLength=128)];
%     lgraph = layerGraph(layer);
%     outputName = lgraph.Layers(end).Name;
% 
%     for k = 1:parameters.numBlocks
% 
%         lgraph = addLayers(lgraph, convolution3dLayer([min(parameters.filterSize, ceil(spatialDim/4^(k-1))) parameters.filterSize parameters.numChannels], parameters.numFilters,...
%             PaddingValue="replicate", Padding="same", Name="conv_" + k));
% 
%         layers = [
%             batchNormalizationLayer(Name="bn_" + k)
%             swishLayer(Name="swish_" + k) %maxPooling1dLayer(2, Stride=2, Name="pooling_"+k)
%             averagePooling3dLayer([4, 1, 1], Name="avg" + k, Stride=[4, 1, 1])
%             spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_"+k)
%             ];
% 
%         lgraph = addLayers(lgraph, layers);
%         lgraph = connectLayers(lgraph, "conv_" + k, "bn_" + k);
%         lgraph = connectLayers(lgraph, outputName, "conv_" + k);
%         outputName = "dropout_" + k;
%     end
% 
%     lgraph= removeLayers(lgraph, "dropout_"+k);
% 
%     outputName = lgraph.Layers(end).Name;
%     layers = [
%             globalAveragePooling2dLayer(Name='avg_layer')
%             flattenLayer(Name="flatten")
%             fullyConnectedLayer(parameters.numLabels,'Name','fc')
%             softmaxLayer
%             classificationLayer];                
% 
%     lgraph = addLayers(lgraph, layers);
%     lgraph = connectLayers(lgraph, outputName, 'avg_layer');
%     analyzeNetwork(lgraph)
% end