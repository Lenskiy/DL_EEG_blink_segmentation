function dlnet = constructCNNRNNs(parameters)

    layers  = [sequenceInputLayer(parameters.numChannels, Name="sequenceInputLayer", MinLength=128)];%, cbt2sscbLayer(Name="t2s")]; % Normalization removed 2023.07
    lgraph = layerGraph(layers);
    
    outputName = layers(end).Name;
    
    % Add convolutional layers
    for k = 1:parameters.numBlocks
        layers = [ %convolution2dLayer([parameters.filterSize 1], parameters.numFilters, Padding="same", Name="conv_" + k)
            convolution1dLayer(parameters.filterSize, parameters.numFilters, Padding="same", Name="conv_" + k)
            layerNormalizationLayer(Name="ln_" + k)
            swishLayer(Name="swish_" + k)
            spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_"+k)
            ];

        lgraph = addLayers(lgraph, layers);
        lgraph = connectLayers(lgraph, outputName, "conv_" + k);
        outputName = "dropout_" + k;
    end
   
    layers = [%sscb2cbtLayer(Name="s2t")
            flattenLayer(Name="fl")];

    lgraph = addLayers(lgraph, layers);
    %lgraph = connectLayers(lgraph, outputName, "s2t");
    lgraph = connectLayers(lgraph, outputName, "fl");

    outputName = layers(end).Name;

    % Add reccurent layers 
    for k = 1:parameters.numRBlocks
        layerOrLayerGraph = eval(parameters.typeOfUnit + "(" + num2str((parameters.numUnits/2^(k-1)) + "," + "Name=['rnn_', num2str(" + k + ")])"));

        if(class(layerOrLayerGraph) == "nnet.cnn.LayerGraph")
            lgraph = mergeNetworkGraphs(lgraph, layerOrLayerGraph);
            lgraph = connectLayers(lgraph, outputName, "rnn_"+k+"_input");
        else
            lgraph = addLayers(lgraph,layerOrLayerGraph);
            lgraph = connectLayers(lgraph, outputName, "rnn_"+k);
        end

        layers = [dropoutLayer(parameters.dropoutFactor, Name="dropout_recurrent_" + k) %layerNormalizationLayer(Name="layernorm_" + i)
                  ];

        lgraph = addLayers(lgraph,layers);
        lgraph = connectLayers(lgraph, "rnn_"+k, layers(end).Name);
        
        outputName = "dropout_recurrent_" + k;


        % layers = [eval(parameters.typeOfUnit + "(" + num2str((parameters.numUnits/2^(k-1)) + "," + "Name=['rnn_', num2str(" + k + ")])"))
        %           dropoutLayer(parameters.dropoutFactor, Name="dropout_recurrent_" + k) %layerNormalizationLayer(Name="layernorm_" + i)
        %           ];
        % lgraph = addLayers(lgraph,layers);
        % lgraph = connectLayers(lgraph, outputName, "rnn_" + k);
        % 
        % outputName = "dropout_recurrent_" + k;
    end

    lgraph = removeLayers(lgraph, "dropout_recurrent_" + k);

    layers = [fullyConnectedLayer(parameters.numLabels,'Name','fc')
              softmaxLayer];    

    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, "rnn_" + k, "fc");
    dlnet = dlnetwork(lgraph);
end


% function lgraph = constructCNNRNNs(parameters)
% 
%     layers  = [sequenceInputLayer(parameters.numChannels, Name="sequenceInputLayer", MinLength=128), cbt2sscbLayer(Name="t2s")]; % Normalization removed 2023.07
%     lgraph = layerGraph(layers);
% 
%     outputName = layers(end).Name;
% 
%     % Add convolutional layers
%     for k = 1:parameters.numBlocks
%         layers = [
%             convolution2dLayer([parameters.filterSize 1], parameters.numFilters, Padding="same", Name="conv_" + k)
%             layerNormalizationLayer(Name="ln_" + k)
%             swishLayer(Name="swish_" + k)
%             spatialDropoutLayer(parameters.dropoutFactor, Name="dropout_"+k)
%             ];
% 
%         lgraph = addLayers(lgraph, layers);
%         lgraph = connectLayers(lgraph, outputName, "conv_" + k);
%         outputName = "dropout_" + k;
%     end
% 
%     layers = [
%             sscb2cbtLayer(Name="s2t")
%             flattenLayer(Name="fl")];
% 
%     lgraph = addLayers(lgraph, layers);
%     lgraph = connectLayers(lgraph, outputName, "s2t");
% 
%     outputName = layers(end).Name;
% 
%     % Add reccurent layers 
%     for k = 1:parameters.numRBlocks
%         layers = [eval(parameters.typeOfUnit + "(" + num2str((parameters.numUnits/2^(k-1)) + "," + "Name=['rnn_', num2str(" + k + ")])"))
%                   dropoutLayer(parameters.dropoutFactor, Name="dropout_recurrent_" + k) %layerNormalizationLayer(Name="layernorm_" + i)
%                   ];
%         lgraph = addLayers(lgraph,layers);
%         lgraph = connectLayers(lgraph, outputName, "rnn_" + k);
% 
%         outputName = "dropout_recurrent_" + k;
%     end
% 
%     lgraph = removeLayers(lgraph, "dropout_recurrent_" + k);
% 
%     layers = [fullyConnectedLayer(parameters.numLabels,'Name','fc')
%               softmaxLayer
%               classificationLayer];    
% 
%     lgraph = addLayers(lgraph, layers);
%     lgraph = connectLayers(lgraph, "rnn_" + k, "fc");
% end