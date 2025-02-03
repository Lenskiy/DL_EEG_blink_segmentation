function dlnet = constructRNN(parameters)
    layer  = [sequenceInputLayer(parameters.numChannels, Name='sequenceInputLayer')];

    lgraph = layerGraph(layer);
    
    outputName = lgraph.Layers(end).Name;
    
    for k = 1:parameters.numRBlocks
        layerOrLayerGraph = eval(parameters.typeOfUnit + "(" + num2str((parameters.numUnits/2^(k-1)) + "," + "Name=['rnn_', num2str(" + k + ")])"));
        
        if(class(layerOrLayerGraph) == "nnet.cnn.LayerGraph")
            lgraph = mergeNetworkGraphs(lgraph, layerOrLayerGraph);
            lgraph = connectLayers(lgraph, outputName, "rnn_"+k+"_input");
        else
            lgraph = addLayers(lgraph,layerOrLayerGraph);
            lgraph = connectLayers(lgraph, outputName, "rnn_"+k);
        end
        

        layers = [dropoutLayer(parameters.dropoutFactor, Name="dropout_" + k) %layerNormalizationLayer(Name="layernorm_" + i)
                  ];

        lgraph = addLayers(lgraph,layers);
        lgraph = connectLayers(lgraph, "rnn_"+k, layers(end).Name);
        
        outputName = "dropout_" + k;
    end

    lgraph = removeLayers(lgraph, "dropout_" + k);

    layers = [fullyConnectedLayer(parameters.numLabels,'Name','fc')
              softmaxLayer];                
    
    lgraph = addLayers(lgraph, layers);
    lgraph = connectLayers(lgraph, "rnn_" + k, "fc");
    dlnet  = dlnetwork(lgraph);
end

% 
% function lgraph = constructRNN(parameters)
%     layer  = [sequenceInputLayer(parameters.numChannels, Name='sequenceInputLayer')];
% 
%     lgraph = layerGraph(layer);
% 
%     outputName = lgraph.Layers(end).Name;
% 
%     for k = 1:parameters.numRBlocks
%         layers = [eval(parameters.typeOfUnit + "(" + num2str((parameters.numUnits/2^(k-1)) + "," + "Name=['rnn_', num2str(" + k + ")])"))
%                   dropoutLayer(parameters.dropoutFactor, Name="dropout_" + k) %layerNormalizationLayer(Name="layernorm_" + i)
%                   ];
%         lgraph = addLayers(lgraph,layers);
%         lgraph = connectLayers(lgraph, outputName, "rnn_"+k);
% 
%         outputName = "dropout_" + k;
%     end
% 
%     lgraph = removeLayers(lgraph, "dropout_" + k);
% 
%     layers = [fullyConnectedLayer(parameters.numLabels,'Name','fc')
%               softmaxLayer
%               classificationLayer];                
% 
%     lgraph = addLayers(lgraph, layers);
%     lgraph = connectLayers(lgraph, "rnn_" + k, "fc");
% end